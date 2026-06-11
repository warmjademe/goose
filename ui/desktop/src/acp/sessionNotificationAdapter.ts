import type { GooseSessionNotification_unstable } from '@aaif/goose-sdk';
import type { ContentBlock, SessionNotification } from '@agentclientprotocol/sdk';
import type { Message, MessageContent, TokenState } from '../api';

export type AcpChatUpdate =
  | { type: 'messages'; messages: Message[] }
  | { type: 'tokenState'; tokenState: Partial<TokenState> }
  | { type: 'sessionInfo'; name?: string };

interface AdapterState {
  messages: Message[];
}

interface GooseMessageMeta {
  messageId?: string;
  created?: number;
}

const DEFAULT_VISIBLE_MESSAGE_METADATA: Message['metadata'] = {
  userVisible: true,
  agentVisible: true,
};

export interface AcpSessionNotificationAdapter {
  apply(notification: SessionNotification): AcpChatUpdate[];
  applyGoose(notification: GooseSessionNotification_unstable): AcpChatUpdate[];
  getMessages(): Message[];
}

export function createAcpSessionNotificationAdapter(
  initialMessages: Message[] = []
): AcpSessionNotificationAdapter {
  const state: AdapterState = {
    messages: initialMessages.map(cloneMessage),
  };

  return {
    apply(notification) {
      return applyAcpSessionNotification(state, notification);
    },
    applyGoose(notification) {
      return applyGooseSessionNotification(notification);
    },
    getMessages() {
      return state.messages.map(cloneMessage);
    },
  };
}

function applyAcpSessionNotification(
  state: AdapterState,
  notification: SessionNotification
): AcpChatUpdate[] {
  const update = notification.update;

  switch (update.sessionUpdate) {
    case 'user_message_chunk':
      return applyContentChunk(state, 'user', update);
    case 'agent_message_chunk':
      return applyContentChunk(state, 'assistant', update);
    case 'agent_thought_chunk':
      return applyThoughtChunk(state, update);
    case 'session_info_update':
      return [
        {
          type: 'sessionInfo',
          ...(update.title ? { name: update.title } : {}),
        },
      ];
    case 'usage_update':
      return [];
    default:
      return [];
  }
}

function applyGooseSessionNotification(
  notification: GooseSessionNotification_unstable
): AcpChatUpdate[] {
  const update = notification.update;

  if (update.sessionUpdate !== 'usage_update') {
    return [];
  }

  return [
    {
      type: 'tokenState',
      tokenState: {
        totalTokens: update.used,
        accumulatedInputTokens: update.accumulatedInputTokens,
        accumulatedOutputTokens: update.accumulatedOutputTokens,
        accumulatedTotalTokens: update.accumulatedInputTokens + update.accumulatedOutputTokens,
        ...(update.accumulatedCost !== undefined
          ? { accumulatedCost: update.accumulatedCost }
          : {}),
      },
    },
  ];
}

function applyContentChunk(
  state: AdapterState,
  role: Message['role'],
  update: Extract<
    SessionNotification['update'],
    { sessionUpdate: 'user_message_chunk' | 'agent_message_chunk' }
  >
): AcpChatUpdate[] {
  const content = messageContentFromAcpContentBlock(update.content);
  if (!content) {
    return [];
  }

  const gooseMeta = getGooseMessageMeta(update);
  const messageId = update.messageId ?? gooseMeta.messageId;
  const existing = findMessageForChunk(state, role, messageId, gooseMeta.created);

  if (existing) {
    const lastContent = existing.content[existing.content.length - 1];
    if (lastContent?.type === 'text' && content.type === 'text') {
      lastContent.text = mergeTextChunk(lastContent.text, content.text);
    } else if (content.type === 'image' && hasImageContent(existing, content)) {
      return [{ type: 'messages', messages: state.messages.map(cloneMessage) }];
    } else {
      existing.content.push(content);
    }
  } else {
    state.messages.push({
      ...(messageId ? { id: messageId } : {}),
      role,
      created: gooseMeta.created ?? Math.floor(Date.now() / 1000),
      content: [content],
      metadata: { ...DEFAULT_VISIBLE_MESSAGE_METADATA },
    });
  }

  return [{ type: 'messages', messages: state.messages.map(cloneMessage) }];
}

function applyThoughtChunk(
  state: AdapterState,
  update: Extract<SessionNotification['update'], { sessionUpdate: 'agent_thought_chunk' }>
): AcpChatUpdate[] {
  if (update.content.type !== 'text') {
    return [];
  }

  const gooseMeta = getGooseMessageMeta(update);
  const messageId = update.messageId ?? gooseMeta.messageId;
  const existing = findMessageForChunk(state, 'assistant', messageId, gooseMeta.created);

  if (existing) {
    const lastContent = existing.content[existing.content.length - 1];
    if (lastContent?.type === 'thinking') {
      lastContent.thinking += update.content.text;
    } else {
      existing.content.push({ type: 'thinking', thinking: update.content.text, signature: '' });
    }
  } else {
    state.messages.push({
      ...(messageId ? { id: messageId } : {}),
      role: 'assistant',
      created: gooseMeta.created ?? Math.floor(Date.now() / 1000),
      content: [{ type: 'thinking', thinking: update.content.text, signature: '' }],
      metadata: { ...DEFAULT_VISIBLE_MESSAGE_METADATA },
    });
  }

  return [{ type: 'messages', messages: state.messages.map(cloneMessage) }];
}

function messageContentFromAcpContentBlock(content: ContentBlock): MessageContent | undefined {
  switch (content.type) {
    case 'text':
      return {
        type: 'text',
        text: content.text,
        ...(content._meta ? { _meta: content._meta } : {}),
        ...(content.annotations ? { annotations: content.annotations } : {}),
      };
    case 'image':
      return {
        type: 'image',
        data: content.data,
        mimeType: content.mimeType,
        ...(content._meta ? { _meta: content._meta } : {}),
        ...(content.annotations ? { annotations: content.annotations } : {}),
      };
    default:
      return undefined;
  }
}

function getGooseMessageMeta(update: { _meta?: unknown }): GooseMessageMeta {
  if (!isRecord(update._meta)) {
    return {};
  }

  const goose = update._meta.goose;
  if (!isRecord(goose)) {
    return {};
  }

  return {
    created: typeof goose.created === 'number' ? goose.created : undefined,
    messageId: typeof goose.messageId === 'string' ? goose.messageId : undefined,
  };
}

function findMessageForChunk(
  state: AdapterState,
  role: Message['role'],
  messageId: string | undefined,
  created: number | undefined
): Message | undefined {
  if (!messageId) {
    return lastMergeableMessageWithRole(state, role);
  }

  const existing = state.messages.find(
    (message) => message.id === messageId && message.role === role
  );
  if (existing) {
    return existing;
  }

  const pending = lastMergeableMessageWithRole(state, role);
  if (pending && !pending.id) {
    pending.id = messageId;
    pending.created = created ?? pending.created;
    return pending;
  }

  return undefined;
}

function lastMergeableMessageWithRole(
  state: AdapterState,
  role: Message['role']
): Message | undefined {
  const lastMessage = state.messages[state.messages.length - 1];
  if (lastMessage?.role !== role || lastMessage.metadata.agentVisible === false) {
    return undefined;
  }
  return lastMessage;
}

function mergeTextChunk(existing: string, incoming: string): string {
  if (!incoming || incoming === existing || existing.endsWith(incoming)) {
    return existing;
  }

  if (!existing || incoming.startsWith(existing)) {
    return incoming;
  }

  return existing + incoming;
}

function hasImageContent(message: Message, image: Extract<MessageContent, { type: 'image' }>) {
  return message.content.some(
    (content) =>
      content.type === 'image' && content.data === image.data && content.mimeType === image.mimeType
  );
}

function cloneMessage(message: Message): Message {
  return {
    ...message,
    content: message.content.map((content) => ({ ...content })),
    metadata: { ...message.metadata },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

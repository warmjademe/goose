import { invoke } from "@tauri-apps/api/core";
import type { SourceEntry } from "@aaif/goose-sdk";
import { getClient } from "@/shared/api/acpConnection";
import type {
  Persona,
  CreatePersonaRequest,
  UpdatePersonaRequest,
  Avatar,
} from "@/shared/types/agents";

const AGENT_SOURCE_TYPE = "agent" as const;
const AGENT_DESCRIPTION = "Agent";

type AgentSourceMetadata = {
  provider?: string;
  model?: string;
  avatar?: string;
};

type AgentSourceEntry = SourceEntry & {
  type: typeof AGENT_SOURCE_TYPE;
  metadata?: AgentSourceMetadata | null;
};

function isAgentSource(source: SourceEntry): source is AgentSourceEntry {
  return source.type === AGENT_SOURCE_TYPE;
}

function avatarToMetadata(
  avatar: Avatar | null | undefined,
): string | undefined {
  if (!avatar) return undefined;
  return avatar.value;
}

function metadataToAvatar(value: string | undefined): Avatar | null {
  if (!value) return null;
  return { type: "url", value };
}

function personaMetadata(
  request: CreatePersonaRequest | UpdatePersonaRequest,
): AgentSourceMetadata | undefined {
  const metadata: AgentSourceMetadata = {};
  if (request.provider) metadata.provider = request.provider;
  if (request.model) metadata.model = request.model;
  const avatar = avatarToMetadata(request.avatar);
  if (avatar) metadata.avatar = avatar;
  return Object.keys(metadata).length > 0 ? metadata : undefined;
}

function toPersona(source: AgentSourceEntry): Persona {
  const writable = source.writable !== false;
  return {
    id: source.path,
    displayName: source.name,
    avatar: metadataToAvatar(source.metadata?.avatar),
    systemPrompt: source.content,
    provider: source.metadata?.provider,
    model: source.metadata?.model,
    isBuiltin: !writable,
    isFromDisk: writable,
    writable,
    createdAt: "",
    updatedAt: "",
  };
}

async function listAgentSources(): Promise<AgentSourceEntry[]> {
  const client = await getClient();
  const response = await client.goose.GooseSourcesList({
    type: AGENT_SOURCE_TYPE,
  });
  return response.sources.filter(isAgentSource);
}

async function getAgentSource(id: string): Promise<AgentSourceEntry> {
  const source = (await listAgentSources()).find(
    (source) => source.path === id,
  );
  if (!source) {
    throw new Error(`Agent '${id}' not found`);
  }
  return source;
}

export async function listPersonas(): Promise<Persona[]> {
  return (await listAgentSources()).map(toPersona);
}

export async function createPersona(
  request: CreatePersonaRequest,
): Promise<Persona> {
  const client = await getClient();
  const response = await client.goose.GooseSourcesCreate({
    type: AGENT_SOURCE_TYPE,
    name: request.displayName,
    description: AGENT_DESCRIPTION,
    content: request.systemPrompt,
    metadata: personaMetadata(request),
    global: true,
  });

  if (!isAgentSource(response.source)) {
    throw new Error(`Unexpected source type returned: ${response.source.type}`);
  }

  return toPersona(response.source);
}

export async function updatePersona(
  id: string,
  request: UpdatePersonaRequest,
): Promise<Persona> {
  const existing = await getAgentSource(id);
  const client = await getClient();
  const merged: CreatePersonaRequest = {
    displayName: request.displayName ?? existing.name,
    avatar:
      request.avatar === undefined
        ? metadataToAvatar(existing.metadata?.avatar)
        : request.avatar,
    systemPrompt: request.systemPrompt ?? existing.content,
    provider: request.provider ?? existing.metadata?.provider,
    model: request.model ?? existing.metadata?.model,
  };
  const response = await client.goose.GooseSourcesUpdate({
    type: AGENT_SOURCE_TYPE,
    path: id,
    name: merged.displayName,
    description: existing.description || AGENT_DESCRIPTION,
    content: merged.systemPrompt,
    metadata: personaMetadata(merged),
  });

  if (!isAgentSource(response.source)) {
    throw new Error(`Unexpected source type returned: ${response.source.type}`);
  }

  return toPersona(response.source);
}

export async function deletePersona(id: string): Promise<void> {
  const client = await getClient();
  await client.goose.GooseSourcesDelete({
    type: AGENT_SOURCE_TYPE,
    path: id,
  });
}

export async function refreshPersonas(): Promise<Persona[]> {
  return listPersonas();
}

export interface ExportResult {
  json: string;
  suggestedFilename: string;
}

export async function exportPersona(id: string): Promise<ExportResult> {
  const client = await getClient();
  const response = await client.goose.GooseSourcesExport({
    type: AGENT_SOURCE_TYPE,
    path: id,
  });
  return { json: response.json, suggestedFilename: response.filename };
}

export async function importPersonas(
  fileBytes: number[],
  fileName: string,
): Promise<Persona[]> {
  if (
    !fileName.endsWith(".agent.json") &&
    !fileName.endsWith(".persona.json") &&
    !fileName.endsWith(".json")
  ) {
    throw new Error(
      "File must have a .agent.json, .persona.json, or .json extension",
    );
  }

  const raw = new TextDecoder().decode(new Uint8Array(fileBytes));
  const parsed = JSON.parse(raw) as Record<string, unknown>;
  const data =
    parsed.type === AGENT_SOURCE_TYPE
      ? raw
      : JSON.stringify({
          version: parsed.version ?? 1,
          type: AGENT_SOURCE_TYPE,
          name: parsed.displayName ?? parsed.name,
          description: AGENT_DESCRIPTION,
          content:
            parsed.systemPrompt ?? parsed.content ?? parsed.instructions ?? "",
          metadata: {
            provider: parsed.provider,
            model: parsed.model,
            avatar:
              typeof parsed.avatar === "string"
                ? parsed.avatar
                : typeof parsed.avatar === "object" &&
                    parsed.avatar !== null &&
                    "value" in parsed.avatar
                  ? (parsed.avatar as { value?: unknown }).value
                  : undefined,
          },
        });

  const client = await getClient();
  const response = await client.goose.GooseSourcesImport({
    data,
    global: true,
  });

  return response.sources.filter(isAgentSource).map(toPersona);
}

export interface ImportFileReadResult {
  fileBytes: number[];
  fileName: string;
}

export async function readImportPersonaFile(
  sourcePath: string,
): Promise<ImportFileReadResult> {
  return invoke("read_import_persona_file", { sourcePath });
}

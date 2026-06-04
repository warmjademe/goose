use crate::base::{Error, Provider, StreamingRequest, StreamingResponse};

struct Anthropic;

impl Provider for Anthropic {
    async fn stream(req: StreamingRequest) -> Result<StreamingResponse, Error> {
        todo!()
    }
}

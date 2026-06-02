# Canonical Model Registry

`goose-models-db` packages Goose's bundled canonical model metadata and model-name normalization helpers.

The registry provides a unified view of model metadata across providers, including:

- canonical model IDs
- context and output token limits
- pricing metadata
- modality and capability metadata
- provider/model name normalization

For example, provider-specific names such as `claude-3-5-sonnet-20241022` can be normalized to canonical IDs such as `anthropic/claude-3.5-sonnet`.

## Updating bundled data

The generator fetches model metadata from `models.dev` and updates the bundled JSON files in `src/canonical/data`:

```bash
cargo run -p goose-models-db --bin build_canonical_models
```

Generated files:

- `src/canonical/data/canonical_models.json`
- `src/canonical/data/provider_metadata.json`

The crate intentionally only packages the registry and its generator. Provider-specific live API validation belongs outside this crate.

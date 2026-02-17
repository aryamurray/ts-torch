# Contributing

Thanks for contributing to ts-torch.

## Development Setup

```bash
bun install
bun run setup
```

## Common Commands

```bash
bun run build
bun run test
bun run check
```

## Pull Request Guidelines

- Keep PRs focused and small when possible
- Add or update tests for behavior changes
- Ensure `bun run check` passes before opening a PR
- Include a changeset for publishable package changes

## Code Style

- TypeScript strict mode
- Formatting via `oxfmt`
- Linting via `oxlint`
- Use `.js` extensions in TypeScript imports for ESM outputs

## Native Code Changes

For `@ts-torch/core/native` updates:

- Verify local native builds with `bun run setup`
- Validate tests that exercise FFI paths
- Confirm platform packaging output is still copied to `packages/@ts-torch-platform/*/lib`

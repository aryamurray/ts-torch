# Changesets

When making changes that should be released, run:
```bash
bun changeset
```

## Pre-1.0 Versioning (Current)

While in 0.x.y, we use semver for unstable APIs:

- **patch** (0.1.0 → 0.1.1): Bug fixes, docs, internal refactors
- **minor** (0.1.0 → 0.2.0): New features AND breaking changes

Do NOT select "major" until we're ready for v1.0.0 stable release.

## Post-1.0 Versioning (Future)

Once stable:
- **patch**: Bug fixes
- **minor**: New features (backwards compatible)
- **major**: Breaking changes

# YAML Formatting

This project uses `yamlfmt` for consistent YAML formatting and `yamllint` for validation.

## Configuration

- **yamlfmt configuration**: `.yamlfmt`
- **yamllint configuration**: `.yamllint`
- **Pre-commit hooks**: Automatically format and validate YAML files

## Usage

### Manual Formatting

Format all YAML files:

```bash
yamlfmt .
```

Check YAML formatting and style:

```bash
yamllint .
```

### Combined Script

Use the convenience script to format and validate:

```bash
uv run format-yaml
```

Or run the Python script directly:

```bash
python utils/format_yaml.py
```

### Pre-commit Integration

YAML files are automatically formatted and validated during git commits via pre-commit hooks.

## Standards

The yamlfmt configuration ensures:

- ✅ Document start markers (`---`) are added to all YAML files
- ✅ Proper comment spacing (minimum 2 spaces before inline comments)
- ✅ Consistent indentation (2 spaces)
- ✅ Consistent line endings (LF)

The yamllint configuration validates:

- ✅ Proper YAML syntax
- ✅ Document structure compliance
- ✅ Comment formatting
- ✅ Indentation consistency
- ⚠️ GitHub Actions truthy values (warns but doesn't fail)

## File Coverage

The configuration applies to:

- `*.yml` and `*.yaml` files in the project root
- `.github/workflows/*.yml` (GitHub Actions)
- `examples/*.yaml` (Configuration examples)
- `config/**/*.yaml` (Configuration files)

Excluded directories:

- `.venv/`, `node_modules/`, `.git/`, `.pytest_cache/`

## Troubleshooting

If you encounter yamllint warnings after running yamlfmt, check:

1. **Document start markers**: All YAML files should start with `---`
1. **Comment spacing**: Inline comments need at least 2 spaces before `#`
1. **GitHub Actions**: The truthy value warnings are expected (e.g., `on: push`)

The formatting script will exit with status 1 if any yamllint issues remain after formatting.

## Script Options

The `format-yaml` script supports several options:

```bash
# Format and validate (default behavior)
uv run format-yaml

# Only validate, don't format
uv run format-yaml --check-only

# Format and validate (explicit)
uv run format-yaml --fix
```

Similar to the `lint` script, the default behavior is to apply fixes unless `--check-only` is specified.

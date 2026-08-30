# MON

Minimum Operational Network

## Install uv

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install Dependencies

```sh
uv sync
```

## Run Tests

```sh
uv run pytest
```

## Dependency Updates

```sh
uv add <package>
```

Dependency changes update pyproject.toml and uv.lock.

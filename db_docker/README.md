# Local database

A Postgres in Docker for seqtante runs and dry runs, with the tables seqtante writes to and no data. Local
only: the login, `seqtante` / `seqtante`, is not a secret.

| | |
|---|---|
| Container | `seqtante-open-db` |
| Host port | 5436 |
| Database | `seqtante` |
| Config | `database.ini`, section `[seqtante]`, schema `autocal` |

## Start

Start Docker Desktop first, then, from the repository root:

```
cd db_docker && docker compose up -d && cd ..
uv run python -m db_docker.create_tables
```

`create_tables` creates the `calibration_run` and `measurements` tables from the installed qililab's own
models, so they always match the qililab seqtante runs with. Tables that exist are left alone, so it is
safe to run again after a qililab upgrade. The dry run calls it itself.

## Use it from a calibration tree

```yaml
storage:
  root_dir: data
  db_manager:
    path: db_docker/database.ini
    database_name: seqtante
```

The schema comes from the section. The `.h5` files go to `data/measurements/`, relative to where seqtante
runs, which `.gitignore` already ignores.

## Look inside

```
docker exec -it seqtante-open-db psql -U seqtante -d seqtante
```

## Stop and reset

```
cd db_docker
docker compose stop        # pause, keep the data
docker compose down        # remove the container, keep the data
docker compose down -v     # remove the container and the data; the tables are created again on next use
```

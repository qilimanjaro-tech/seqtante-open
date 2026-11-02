Added the `database_schema` key to `storage.db_manager`, so Seqtante-Open can run on the measurements database as well as on the autocalibration one.

- **database_schema** : `"autocal" | "measurements"`, optional.
  Database the manager is connected to, passed to qililab's `get_db_manager`. If it is not set, the `database_schema` key of the `.ini` section is used. If both are set and differ, this one is used and qililab raises a warning.

- The experiment functions no longer pass `autocalibration` to `StreamArray`, so each measurement is saved in the table of the connected database, following its `database_schema`.
- `FittingClass` loads measurements with `load_entry_by_id`, so fits work on both schemas, and gained a `platform` cached property that builds the `Platform` from the runcard stored with the measurement. It raises a `ValueError` if the measurement has no platform stored.

## **Example**
```yml
storage:
  root_dir: "/path/to/output/directory"
  db_manager:
    path: "~/database.ini"
    database_name: "calibration"
    database_schema: "autocal"
```

# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create the tables seqtante writes to, empty, from the installed qililab's own models.

    uv run python -m db_docker.create_tables
    uv run python -m db_docker.create_tables --path my_database.ini --section my_section

The section's ``database_schema`` picks the tables: ``autocal`` (the default) or ``measurements``.
Tables that already exist are left as they are.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qililab.result.database import CalibrationRun, SequenceRun, get_db_manager

DATABASE_INI = Path(__file__).resolve().parent / "database.ini"
SECTION = "seqtante"


def create_tables(path: str | Path = DATABASE_INI, section: str = SECTION) -> list[str]:
    """Create the tables of the section's ``database_schema`` that do not exist yet; return the tables it has.

    Raises:
        sqlalchemy.exc.OperationalError: if the database cannot be reached.
        ValueError: if the section's schema is neither ``autocal`` nor ``measurements``.
    """
    db_manager = get_db_manager(path=str(path), database_name=section)
    models = {"autocal": CalibrationRun, "measurements": SequenceRun}
    if db_manager.database_schema not in models:
        raise ValueError(f"seqtante writes 'autocal' or 'measurements', not '{db_manager.database_schema}'")
    metadata = models[db_manager.database_schema].metadata
    metadata.create_all(db_manager.engine)
    return sorted(metadata.tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=str(DATABASE_INI))
    parser.add_argument("--section", default=SECTION)
    args = parser.parse_args()
    tables = create_tables(args.path, args.section)
    sys.stdout.write(f"[{args.section}] has the tables {', '.join(tables)}\n")

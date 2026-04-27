"""Allow ``python -m geotempfill`` to invoke the CLI."""

from .cli import main

raise SystemExit(main())

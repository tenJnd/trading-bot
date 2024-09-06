from logging.config import fileConfig

from alembic import context

from src.model import trader_database
from src.model.turtle_model import Base, SCHEMA

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def include_name(name, type_, parent_names):
    if type_ == "schema":
        # schemas restriction
        return name in [f"{SCHEMA}"]
    if type_ == "table" and parent_names["schema_name"] is None:
        return f"{name}" in target_metadata.tables
    if type_ == "table":
        # table restriction
        schema_name = parent_names["schema_name"] if parent_names["schema_name"] is not None else "public"
        return (
                f"""{schema_name}.{name}""" in
                target_metadata.tables
        )
    # elif type_ == "index":
    #     schema_name = parent_names["schema_name"] if parent_names["schema_name"] is not None else "public"
    #     print(f"mame index--- {schema_name}.{name}------------------------------------------------------------------")
    #     name = f"{schema_name}.{name}"
    #     return True
    return True


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = trader_database.engine.url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,  # allows alembic to work with all of our db schemas
        include_name=include_name  # applies schema and table restrictions
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = trader_database.engine

    with connectable.connect() as connection:
        # print("Target metadata: ", target_metadata.tables)
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=target_metadata.schema,
            include_schemas=True,  # allows alembic to work with all of our db schemas
            include_name=include_name  # applies schema and table restrictions
        )

        with context.begin_transaction():
            # allows correct cross schema foreign keys creation
            context.execute(f'SET search_path TO {SCHEMA}')
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

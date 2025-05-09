"""add candle timestamp into agent action

Revision ID: 8b3470275683
Revises: 1440dfcfb9b5
Create Date: 2024-11-13 18:24:12.154102

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '8b3470275683'
down_revision = '1440dfcfb9b5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('agent_actions', sa.Column('candle_timestamp', sa.BigInteger(), nullable=True),
                  schema='turtle_strategy')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('agent_actions', 'candle_timestamp', schema='turtle_strategy')
    # ### end Alembic commands ###

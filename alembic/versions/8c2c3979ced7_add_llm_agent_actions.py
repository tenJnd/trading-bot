"""add llm agent actions

Revision ID: 8c2c3979ced7
Revises: f94b4942e9cc
Create Date: 2024-10-10 14:11:48.673130

"""
import sqlalchemy as sa
import sqlalchemy_utc
from alembic import op

# revision identifiers, used by Alembic.
revision = '8c2c3979ced7'
down_revision = 'f94b4942e9cc'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('agent_actions',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('action', sa.String(), nullable=True),
                    sa.Column('rationale', sa.String(), nullable=True),
                    sa.Column('agent_output', sa.JSON(), nullable=True),
                    sa.Column('strategy_id', sa.Integer(), nullable=True),
                    sa.Column('timestamp_created', sqlalchemy_utc.sqltypes.UtcDateTime(timezone=True), nullable=False),
                    sa.ForeignKeyConstraint(['strategy_id'], ['turtle_strategy.strategy_settings.id'], ),
                    sa.PrimaryKeyConstraint('id'),
                    schema='turtle_strategy'
                    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('agent_actions', schema='turtle_strategy')
    # ### end Alembic commands ###

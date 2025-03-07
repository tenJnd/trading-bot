"""order to llm actions

Revision ID: eb5333507519
Revises: f05e73f4a53a
Create Date: 2024-10-10 16:55:20.869321

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'eb5333507519'
down_revision = 'f05e73f4a53a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('agent_actions', sa.Column('order', sa.JSON(), nullable=True), schema='turtle_strategy')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('agent_actions', 'order', schema='turtle_strategy')
    # ### end Alembic commands ###

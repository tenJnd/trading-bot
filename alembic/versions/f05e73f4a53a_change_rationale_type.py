"""change rationale type

Revision ID: f05e73f4a53a
Revises: 8c2c3979ced7
Create Date: 2024-10-10 15:31:06.231795

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'f05e73f4a53a'
down_revision = '8c2c3979ced7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('agent_actions', 'rationale',
                    existing_type=sa.VARCHAR(),
                    type_=sa.Text(),
                    existing_nullable=True,
                    schema='turtle_strategy')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('agent_actions', 'rationale',
                    existing_type=sa.Text(),
                    type_=sa.VARCHAR(),
                    existing_nullable=True,
                    schema='turtle_strategy')
    # ### end Alembic commands ###
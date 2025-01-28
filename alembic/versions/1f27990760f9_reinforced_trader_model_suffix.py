"""reinforced trader model suffix

Revision ID: 1f27990760f9
Revises: de62e9032565
Create Date: 2025-01-17 10:48:11.982337

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '1f27990760f9'
down_revision = 'de62e9032565'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('episodes_training', sa.Column('model_suffix', sa.String(), nullable=True), schema='turtle_strategy')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('episodes_training', 'model_suffix', schema='turtle_strategy')
    # ### end Alembic commands ###

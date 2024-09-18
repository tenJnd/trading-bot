"""
Revises: a0eb15d5f053
Create Date: 2024-09-18 20:40:06.563567

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '7f536ecd893d'
down_revision = 'a0eb15d5f053'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        'orders',
        sa.Column('atr_period_ratio', sa.Float(), nullable=False, server_default='1.0'),
        schema='turtle_strategy'
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('orders', 'atr_period_ratio', schema='turtle_strategy')
    # ### end Alembic commands ###

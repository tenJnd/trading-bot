"""report daily

Revision ID: deaaf055ac20
Revises: 30e7d119adc1
Create Date: 2024-12-25 09:51:31.307488

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'deaaf055ac20'
down_revision = '30e7d119adc1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('balance_report', sa.Column('date', sa.String(), nullable=True), schema='turtle_strategy')
    op.drop_column('balance_report', 'candle_timestamp', schema='turtle_strategy')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('balance_report', sa.Column('candle_timestamp', sa.VARCHAR(), autoincrement=False, nullable=True),
                  schema='turtle_strategy')
    op.drop_column('balance_report', 'date', schema='turtle_strategy')
    # ### end Alembic commands ###

"""add settings table

Revision ID: c2d5be37ffc3
Revises: 041ab0ceb0ac
Create Date: 2024-09-08 20:53:39.571340

"""
import sqlalchemy as sa
import sqlalchemy_utc
from alembic import op

# revision identifiers, used by Alembic.
revision = 'c2d5be37ffc3'
down_revision = '041ab0ceb0ac'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('strategy_settings',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('exchange_id', sa.String(), nullable=True),
                    sa.Column('ticker', sa.String(), nullable=True),
                    sa.Column('timeframe', sa.String(), nullable=True),
                    sa.Column('buffer_days', sa.Integer(), nullable=True),
                    sa.Column('timestamp_created', sqlalchemy_utc.sqltypes.UtcDateTime(timezone=True), nullable=False),
                    sa.PrimaryKeyConstraint('id'),
                    schema='turtle_strategy'
                    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('strategy_settings', schema='turtle_strategy')
    # ### end Alembic commands ###
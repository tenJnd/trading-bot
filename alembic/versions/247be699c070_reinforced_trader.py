"""reinforced trader

Revision ID: 247be699c070
Revises: deaaf055ac20
Create Date: 2025-01-14 09:42:38.548532

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '247be699c070'
down_revision = 'deaaf055ac20'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('episodes_training',
                    sa.Column('episode_group', sa.String(), nullable=False),
                    sa.Column('balance', sa.Float(), nullable=True),
                    sa.Column('total_reward', sa.Float(), nullable=True),
                    sa.Column('total_profit', sa.Float(), nullable=True),
                    sa.Column('win_trades', sa.Integer(), nullable=True),
                    sa.Column('lost_trades', sa.Integer(), nullable=True),
                    sa.PrimaryKeyConstraint('episode_group'),
                    schema='turtle_strategy'
                    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('episodes_training', schema='turtle_strategy')
    # ### end Alembic commands ###

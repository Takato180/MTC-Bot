# このファイルは、戦略ルールをデータフレームに対して適用し、トレーディングシグナルを生成するためのクラスを定義しています。



import pandas as pd
from pydantic import BaseModel, Field

# Ruleクラスは、各ルールの左辺（列名）、演算子（'>' または '<'）、右辺（定数または他の列名）を表現します。
class Rule(BaseModel):
    lhs: str
    op:  str  # '>' または '<'
    rhs: float | str  # 数値 or 列名


# RuleStrategyクラスは、エントリーとイグジットのルールのリストを持ち、与えられたデータフレームに対して各ルールを評価し、
# シグナル（ポジション）のSeriesを返すメソッド(signal)を提供します。
class RuleStrategy(BaseModel):
    entry: list[Rule]
    exit:  list[Rule]

    def signal(self, df: pd.DataFrame) -> pd.Series:
        def _eval(rule: Rule):
            lhs = df[rule.lhs]
            rhs = df[rule.rhs] if isinstance(rule.rhs, str) else rule.rhs
            return lhs.gt(rhs) if rule.op == '>' else lhs.lt(rhs)

        long_sig  = pd.concat([_eval(r) for r in self.entry], axis=1).all(axis=1)
        flat_sig  = pd.concat([_eval(r) for r in self.exit],  axis=1).all(axis=1)
        position  = long_sig.astype(int).where(~flat_sig).ffill().fillna(0)
        return position

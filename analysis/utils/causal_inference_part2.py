def is_effect_significant(ci_lower: float, ci_upper: float) -> bool:
    """
    判断因果效应是否统计显著

    参数:
        ci_lower: 置信区间下界
        ci_upper: 置信区间上界

    返回:
        is_significant: True如果置信区间不包含0
    """
    return not (ci_lower <= 0 <= ci_upper)


def format_ate_result(ate: float, ci: Tuple[float, float]) -> str:
    """
    格式化ATE结果为可读字符串

    参数:
        ate: 平均处理效应
        ci: 置信区间 (lower, upper)

    返回:
        formatted_string: 格式化的字符串
    """
    is_sig = is_effect_significant(ci[0], ci[1])
    sig_marker = "***" if is_sig else ""

    return f"ATE={ate:.4f} {sig_marker}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]"

    def build_reference_df(self,
                          data: pd.DataFrame,
                          strategy: str = "non_parallel",
                          groupby_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        构建参考数据集（ref_df）用于CTF风格的ATE计算

        参数:
            data: 原始数据集
            strategy: 构建策略
                - "non_parallel": 使用非并行模式作为baseline（推荐）
                - "mean": 使用全局均值
                - "group_mean": 按组计算均值
            groupby_cols: 分组列（仅当strategy="group_mean"时使用）

        返回:
            ref_df: 参考数据集

        示例:
            >>> # 使用非并行模式作为baseline
            >>> ref_df = ci.build_reference_df(data, strategy="non_parallel")
            >>>
            >>> # 使用全局均值
            >>> ref_df = ci.build_reference_df(data, strategy="mean")
        """
        if strategy == "non_parallel":
            # 使用非并行模式作为baseline（推荐用于能耗分析）
            if 'is_parallel' in data.columns:
                ref_df = data[data['is_parallel'] == 0].copy()
                if self.verbose:
                    print(f"  构建ref_df: 非并行模式 (n={len(ref_df)})")
            else:
                warnings.warn("数据中无is_parallel列，使用全部数据")
                ref_df = data.copy()
                if self.verbose:
                    print(f"  构建ref_df: 全部数据 (n={len(ref_df)})")

        elif strategy == "mean":
            # 使用全局均值
            ref_df = data.mean().to_frame().T
            if self.verbose:
                print(f"  构建ref_df: 全局均值 (1行)")

        elif strategy == "group_mean":
            # 按组计算均值
            if groupby_cols is None:
                raise ValueError("group_mean策略需要指定groupby_cols参数")

            # 检查列是否存在
            missing_cols = [col for col in groupby_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"groupby_cols中的列不存在: {missing_cols}")

            ref_df = data.groupby(groupby_cols).mean().reset_index()
            if self.verbose:
                print(f"  构建ref_df: 按{groupby_cols}分组 (n={len(ref_df)})")

        else:
            raise ValueError(f"未知的strategy: {strategy}")

        # 验证ref_df不为空
        if len(ref_df) == 0:
            raise ValueError("构建的ref_df为空，请检查数据和策略")

        return ref_df

    def compute_T0_T1(self,
                     data: pd.DataFrame,
                     treatment: str,
                     strategy: str = "quantile") -> Tuple[float, float]:
        """
        计算T0（控制组值）和T1（处理组值）

        参数:
            data: 数据集
            treatment: 处理变量名
            strategy: 计算策略
                - "quantile": 25/75分位数（推荐，鲁棒性强）
                - "min_max": 最小值/最大值
                - "mean_std": 均值±标准差

        返回:
            (T0, T1): 控制组值和处理组值

        Raises:
            ValueError: 如果T1 <= T0（数据变异性不足）

        示例:
            >>> # 使用分位数策略（推荐）
            >>> T0, T1 = ci.compute_T0_T1(data, 'learning_rate', strategy='quantile')
            >>>
            >>> # 使用最小最大值
            >>> T0, T1 = ci.compute_T0_T1(data, 'learning_rate', strategy='min_max')
        """
        if treatment not in data.columns:
            raise ValueError(f"treatment '{treatment}' 不在数据中")

        if strategy == "quantile":
            # 使用25/75分位数（推荐，更鲁棒）
            T0 = data[treatment].quantile(0.25)
            T1 = data[treatment].quantile(0.75)

        elif strategy == "min_max":
            # 使用最小值/最大值
            T0 = data[treatment].min()
            T1 = data[treatment].max()

        elif strategy == "mean_std":
            # 使用均值±标准差
            mean_val = data[treatment].mean()
            std_val = data[treatment].std()
            T0 = mean_val - std_val
            T1 = mean_val + std_val

        else:
            raise ValueError(f"未知的strategy: {strategy}")

        # 验证T1 > T0
        if T1 <= T0:
            raise ValueError(
                f"T1 ({T1:.4f}) <= T0 ({T0:.4f})。"
                f"数据变异性不足，请检查treatment变量'{treatment}'"
            )

        # 警告：如果差异太小
        diff = T1 - T0
        threshold = 0.1 * data[treatment].std()
        if diff < threshold:
            warnings.warn(
                f"T1-T0差异较小: {diff:.4f} (阈值={threshold:.4f})。"
                f"ATE估计可能不稳定。"
            )

        if self.verbose:
            print(f"  计算T0/T1: {T0:.4f} / {T1:.4f} (策略={strategy}, 差异={diff:.4f})")

        return T0, T1

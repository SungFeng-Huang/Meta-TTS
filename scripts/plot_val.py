import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

df = []
exp_id = "1d34914e07244f87ad10c580ab85656c"
csv_dir = f"output/log/LibriTTS/meta-tts-accent-dev/{exp_id}/csv/"

for i in range(10000, 100001, 10000):
    csv_path = os.path.join(csv_dir, f"validate-{i}.csv")
    df.append(pd.read_csv(csv_path))

df = pd.concat(df, ignore_index=True)

with sns.axes_style("whitegrid"):
    sns.relplot(
        kind="line",
        data=df,
        x="global_step",
        y="Total Loss",
        hue="ft_step",
        legend="full",
        err_style="bars", ci=68,
        col="dset_name", col_wrap=3,
    )

# with sns.axes_style("whitegrid"):
#     sns.relplot(
#         kind="line",
#         data=df[df["dset_name"].values != "LibriTTS/dev-clean"],
#         x="ft_step",
#         y="accent_acc",
#         hue="global_step",
#         legend="full",
#         err_style="bars", ci=68,
#         col="dset_name", col_wrap=3,
#     )

# df = pd.concat([
#     pd.read_csv(os.path.join(csv_dir, "val-10000.csv")),
#     pd.read_csv(os.path.join(csv_dir, "val-100000.csv")),
# ], ignore_index=True)
#
# df = df.melt(
#     id_vars=[k for k in df.keys() if k not in ["DataLoader 0", "DataLoader 1", "DataLoader 2"]],
#     value_vars=["DataLoader 0", "DataLoader 1", "DataLoader 2"],
#     var_name="dset_name", value_name="accent_acc",
# )
# df_oracle = df[df["ft_step"].values == "oracle"]
# df = df[df["ft_step"].values != "oracle"]
# df.apply(pd.to_numeric, errors='ignore')
# with sns.axes_style("whitegrid"):
#     g = sns.relplot(
#         kind="line",
#         data=df,
#         x="ft_step",
#         y="accent_acc",
#         hue="global_step",
#         style="Validate metric",
#         legend="full",
#         err_style="bars", ci=68,
#         col="dset_name", col_wrap=3,
#         # sort=False,
#     )
#     # print(g.get_legend_handles_labels())
#     # for i, ax in enumerate(g.axes.flatten()):
#     #     ax.axhline(
#     #         y=df_oracle[
#     #             df_oracle["Validate metric"].values == "acc"
#     #         ][
#     #             df_oracle["dset_name"].values == f"DataLoader {i}"
#     #         ]["accent_acc"].values[0],
#     #         c='b', ls='-',
#     #     )
#     #     ax.axhline(
#     #         y=df_oracle[
#     #             df_oracle["Validate metric"].values == "prob"
#     #         ][
#     #             df_oracle["dset_name"].values == f"DataLoader {i}"
#     #         ]["accent_acc"].values[0],
#     #         c='b', ls='--',
#     #     )

plt.show()

root_dir = '../'
corpus = 'VCTK'
corpus = 'LibriTTS'

data_dir_dict = dict()
if corpus == 'LibriTTS':
    data_dir_dict.update({
        'real': f'{root_dir}/raw_data/{corpus}/test-clean/',
        'enrollment': f'{root_dir}/raw_data/{corpus}/test-clean/',
    })
    n_speaker = 38
    tsne_pseudo_speaker_list = [3, 15, 25]
else:
    data_dir_dict.update({
        'real': f'{root_dir}/raw_data/{corpus}/all/',
        'enrollment': f'{root_dir}/raw_data/{corpus}/all/',
    })
    n_speaker = 108
    tsne_pseudo_speaker_list = [78, 37, 97]

result_root = f"{root_dir}/output/result/{corpus}"
data_dir_dict.update({
    'recon': f'{result_root}/672c4ace93c04b57a48911549ef0e609/base_emb_vad',
    'base_emb_vad': f'{result_root}/672c4ace93c04b57a48911549ef0e609/base_emb_vad',
    'base_emb_va': f'{result_root}/672c4ace93c04b57a48911549ef0e609/base_emb_va',
    'base_emb_d': f'{result_root}/672c4ace93c04b57a48911549ef0e609/base_emb_d',
    'base_emb': f'{result_root}/672c4ace93c04b57a48911549ef0e609/base_emb',
})
data_dir_dict.update({
    'meta_emb_vad': f'{result_root}/960dba64771045a9b1d4e48dd90b2270/meta_emb_vad',
    'meta_emb_va': f'{result_root}/8a681754f67142d1a06bd1f9cafe3ace/meta_emb_va',
    'meta_emb_d': f'{result_root}/3d7b257fc38245489348ac8972df187e/meta_emb_d',
    'meta_emb': f'{result_root}/4b048a96be444017b978f9e21ea664e6/meta_emb',
})
data_dir_dict.update({
    'base_emb1_vad': f'{result_root}/b3d4b916db01475d94fd690da6f25ae2/base_emb1_vad',
    'base_emb1_va': f'{result_root}/b3d4b916db01475d94fd690da6f25ae2/base_emb1_va',
    'base_emb1_d': f'{result_root}/b3d4b916db01475d94fd690da6f25ae2/base_emb1_d',
    'base_emb1': f'{result_root}/b3d4b916db01475d94fd690da6f25ae2/base_emb1',
})
data_dir_dict.update({
    'meta_emb1_vad': f'{result_root}/8f1d5e4c2db64bfd886d5f981b58974c/meta_emb1_vad',
    'meta_emb1_va': f'{result_root}/76d5bf2e9c044908bf7122d350488cff/meta_emb1_va',
    'meta_emb1_d': f'{result_root}/c0e9e6a6f5984cb28fa05522b830cfec/meta_emb1_d',
    'meta_emb1': f'{result_root}/eaca69ba824b45bfb8d6e1f663bc6c51/meta_emb1',
})
data_dir_dict.update({
    'scratch_encoder': f'{result_root}/064fdd9ccfa94ca190d0dcccead456ce/scratch_encoder',
    'encoder': f'{result_root}/b40400015bac4dfd8a8aaffec7d3db9f/encoder',
    'dvec': f'{result_root}/fdf55e6b33434922b758d034e839f000/dvec',
})
data_dir_dict.update({
    'base_emb_vad-train_clean': f'{result_root}/e8e52fb836e74d3b9e7a27e7b5a7fa95/base_emb_vad-train_clean',
    'base_emb_vad-train_all': f'{result_root}/a7442f4f9af74b8caf2641f6edeca616/base_emb_vad-train_all',
    'meta_emb_vad-train_clean': f'{result_root}/554524d810e44e9e9688e205fb0c04ba/meta_emb_vad-train_clean',
    'meta_emb_vad-train_all': f'{result_root}/ac596d75e85a4b08bac03e86b7fa45ad/meta_emb_vad-train_all',
})
data_dir_dict.update({
    'base_emb_vad-avg_train_spk_emb': f'{result_root}/672c4ace93c04b57a48911549ef0e609/base_emb_vad-avg_train_spk_emb',
    'base_emb_vad-train_clean-avg_train_spk_emb': f'{result_root}/e8e52fb836e74d3b9e7a27e7b5a7fa95/base_emb_vad-train_clean-avg_train_spk_emb',
    'base_emb_vad-train_all-avg_train_spk_emb': f'{result_root}/a7442f4f9af74b8caf2641f6edeca616/base_emb_vad-train_all-avg_train_spk_emb',
    'meta_emb_vad-avg_train_spk_emb': f'{result_root}/960dba64771045a9b1d4e48dd90b2270/meta_emb_vad-avg_train_spk_emb',
    'meta_emb_vad-train_clean-avg_train_spk_emb': f'{result_root}/554524d810e44e9e9688e205fb0c04ba/meta_emb_vad-train_clean-avg_train_spk_emb',
    'meta_emb_vad-train_all-avg_train_spk_emb': f'{result_root}/ac596d75e85a4b08bac03e86b7fa45ad/meta_emb_vad-train_all-avg_train_spk_emb',
})
data_dir_dict.update({
    'base_emb_vad-1_shot': f'{result_root}/672c4ace93c04b57a48911549ef0e609/base_emb_vad-1_shot',
    'meta_emb_vad-1_shot': f'{result_root}/960dba64771045a9b1d4e48dd90b2270/meta_emb_vad-1_shot',
})

n_sample = 16
mode_list = [
    'base_emb_vad',
    'base_emb_va',
    'base_emb_d',
    'base_emb',
    'meta_emb_vad',
    'meta_emb_va',
    'meta_emb_d',
    'meta_emb',
    'base_emb1_vad',
    'base_emb1_va',
    'base_emb1_d',
    'base_emb1',
    'meta_emb1_vad',
    'meta_emb1_va',
    'meta_emb1_d',
    'meta_emb1',
    'scratch_encoder',
    'encoder',
    'dvec',
    'base_emb_vad-train_clean',
    'base_emb_vad-train_all',
    'meta_emb_vad-train_clean',
    'meta_emb_vad-train_all',
    'base_emb_vad-avg_train_spk_emb',
    'base_emb_vad-train_clean-avg_train_spk_emb',
    'base_emb_vad-train_all-avg_train_spk_emb',
    'meta_emb_vad-avg_train_spk_emb',
    'meta_emb_vad-train_clean-avg_train_spk_emb',
    'meta_emb_vad-train_all-avg_train_spk_emb',
    'base_emb_vad-1_shot',
    'meta_emb_vad-1_shot',
]
step_list = [0, 5, 10, 20, 50, 100]
mode_step_list = [
    ('base_emb_va', step_list),
    ('base_emb_d', step_list),
    ('base_emb', step_list),
    ('meta_emb_va', step_list),
    ('meta_emb_d', step_list),
    ('meta_emb', step_list),
    ('base_emb1_vad', step_list),
    ('base_emb1_va', step_list),
    ('base_emb1_d', step_list),
    ('base_emb1', step_list),
    ('meta_emb1_vad', step_list),
    ('meta_emb1_va', step_list),
    ('meta_emb1_d', step_list),
    ('meta_emb1', step_list),
    ('scratch_encoder', [0]),
    ('encoder', [0]),
    ('dvec', [0]),
    ('base_emb_vad', step_list),
    ('base_emb_vad-train_clean', step_list),
    ('base_emb_vad-train_all', step_list),
    ('meta_emb_vad', step_list),
    ('meta_emb_vad-train_clean', step_list),
    ('meta_emb_vad-train_all', step_list),
    ('base_emb_vad-avg_train_spk_emb', step_list),
    ('base_emb_vad-train_clean-avg_train_spk_emb', step_list),
    ('base_emb_vad-train_all-avg_train_spk_emb', step_list),
    ('meta_emb_vad-avg_train_spk_emb', step_list),
    ('meta_emb_vad-train_clean-avg_train_spk_emb', step_list),
    ('meta_emb_vad-train_all-avg_train_spk_emb', step_list),
    ('base_emb_vad-1_shot', [0, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]),
    ('meta_emb_vad-1_shot', [0, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]),
]


# parameters for visualize.py
tsne_mode_list = ['recon', 'base_emb1_vad_step20', 'base_emb_vad_step20', 'meta_emb1_vad_step20', 'meta_emb_vad_step20']
tsne_legend_list = [
    'Reconstructed',
    'Baseline (share emb)',
    'Baseline (emb table)',
    'Meta-TTS (share emb)',
    'Meta-TTS (emb table)',
]
tsne_plot_color_list = ['grey', 'orange', 'red', 'green', 'blue']


# parameters for centroid_similarity.py
centroid_sim_mode_list = ['centroid', 'recon_random','recon'] + mode_list


# parameters for similarity_plot.py
plot_type = 'errorbar'   # ['errorbar', 'box_ver', 'box_hor']
sim_plot_mode_list = [
    'recon', 'recon_random',
    'base_emb1_vad',
    'base_emb_vad',
    'meta_emb1_vad',
    'meta_emb_vad',
]
# length of color_list should be same as len(sim_plot_mode_list)
sim_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
# length of legend_list should be same as len(sim_plot_mode_list)
sim_plot_legend_list = [
    'Same spk', 'Different spk',
    'Baseline (share emb)',
    'Baseline (emb table)',
    'Meta-TTS (share emb)',
    'Meta-TTS (emb table)',
]


#parameters for computing EER in speaker_verification.py
eer_plot_mode_list = [
    'real',
    'recon', 
    'base_emb1_vad',
    'base_emb_vad',
    'meta_emb1_vad',
    'meta_emb_vad',
]
eer_plot_legend_list = [
    'Real',
    'Reconstructed',
    'Baseline (share emb)',
    'Baseline (emb table)',
    'Meta-TTS (share emb)',
    'Meta-TTS (emb table)',
]
eer_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']

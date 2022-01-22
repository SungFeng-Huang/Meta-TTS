root_dir = '../'
corpus = 'VCTK'
# corpus = 'LibriTTS'

data_dir_dict = dict()
if corpus == 'LibriTTS':
    data_dir_dict['real'] = f'{root_dir}/raw_data/{corpus}/test-clean/'
    data_dir_dict['enrollment'] = f'{root_dir}/raw_data/{corpus}/test-clean/'
    n_speaker = 38
    tsne_pseudo_speaker_list = [3, 15, 25]
else:
    data_dir_dict['real'] = f'{root_dir}/raw_data/{corpus}/all/'
    data_dir_dict['enrollment'] = f'{root_dir}/raw_data/{corpus}/all/'
    n_speaker = 108
    tsne_pseudo_speaker_list = [78, 37, 97]
data_dir_dict['recon'] = f'{root_dir}/output/result/{corpus}/672c4ace93c04b57a48911549ef0e609/base_emb_vad'
data_dir_dict['base_emb_vad'] = f'{root_dir}/output/result/{corpus}/672c4ace93c04b57a48911549ef0e609/base_emb_vad'
data_dir_dict['base_emb_va'] = f'{root_dir}/output/result/{corpus}/672c4ace93c04b57a48911549ef0e609/base_emb_va'
data_dir_dict['base_emb_d'] = f'{root_dir}/output/result/{corpus}/672c4ace93c04b57a48911549ef0e609/base_emb_d'
data_dir_dict['base_emb'] = f'{root_dir}/output/result/{corpus}/672c4ace93c04b57a48911549ef0e609/base_emb'
data_dir_dict['meta_emb_vad'] = f'{root_dir}/output/result/{corpus}/960dba64771045a9b1d4e48dd90b2270/meta_emb_vad'
data_dir_dict['meta_emb_va'] = f'{root_dir}/output/result/{corpus}/8a681754f67142d1a06bd1f9cafe3ace/meta_emb_va'
data_dir_dict['meta_emb_d'] = f'{root_dir}/output/result/{corpus}/3d7b257fc38245489348ac8972df187e/meta_emb_d'
data_dir_dict['meta_emb'] = f'{root_dir}/output/result/{corpus}/4b048a96be444017b978f9e21ea664e6/meta_emb'
data_dir_dict['base_emb1_vad'] = f'{root_dir}/output/result/{corpus}/b3d4b916db01475d94fd690da6f25ae2/base_emb1_vad'
data_dir_dict['base_emb1_va'] = f'{root_dir}/output/result/{corpus}/b3d4b916db01475d94fd690da6f25ae2/base_emb1_va'
data_dir_dict['base_emb1_d'] = f'{root_dir}/output/result/{corpus}/b3d4b916db01475d94fd690da6f25ae2/base_emb1_d'
data_dir_dict['base_emb1'] = f'{root_dir}/output/result/{corpus}/b3d4b916db01475d94fd690da6f25ae2/base_emb1'
data_dir_dict['meta_emb1_vad'] = f'{root_dir}/output/result/{corpus}/8f1d5e4c2db64bfd886d5f981b58974c/meta_emb1_vad'
data_dir_dict['meta_emb1_va'] = f'{root_dir}/output/result/{corpus}/76d5bf2e9c044908bf7122d350488cff/meta_emb1_va'
data_dir_dict['meta_emb1_d'] = f'{root_dir}/output/result/{corpus}/c0e9e6a6f5984cb28fa05522b830cfec/meta_emb1_d'
data_dir_dict['meta_emb1'] = f'{root_dir}/output/result/{corpus}/eaca69ba824b45bfb8d6e1f663bc6c51/meta_emb1'
data_dir_dict['scratch_encoder'] = f'{root_dir}/output/result/{corpus}/064fdd9ccfa94ca190d0dcccead456ce/scratch_encoder'
data_dir_dict['encoder'] = f'{root_dir}/output/result/{corpus}/b40400015bac4dfd8a8aaffec7d3db9f/encoder'
data_dir_dict['dvec'] = f'{root_dir}/output/result/{corpus}/fdf55e6b33434922b758d034e839f000/dvec'
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
]
step_list = [0, 5, 10, 20, 50, 100]


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

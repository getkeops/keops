function clean_keops()

[~, build_dir, ~, ~, ~] = default_options();
eval(['!rm -rf ',build_dir,'*']);
eval(['!touch ',build_dir,'.gitkeep']);


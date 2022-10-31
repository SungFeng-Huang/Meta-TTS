for spk in p308
do
  source scripts/learnable_structured_pipeline.sh $spk FT-mask
done

for spk in p226 p231 p236 p254 p257 p259 p272 p273 p274 p276 p277 p292 p298 \
  p314 p326 p334 p341 p343 p374
do
  source scripts/learnable_structured_pipeline.sh $spk joint
done

for spk in p237 p270
do
  source scripts/learnable_structured_pipeline.sh $spk libri_mask-FT
done

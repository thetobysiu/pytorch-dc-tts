import zarr

voice = 'Yennefer'
spec = zarr.open(f'datasets/{voice}/spec.zip', mode='r', synchronizer=zarr.ThreadSynchronizer())
zarr.save_group(f'datasets/{voice}/mels.zip', **spec.mels)
zarr.save_group(f'datasets/{voice}/mags.zip', **spec.mags)

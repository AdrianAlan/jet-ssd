import h5py
import numpy as np

index = 0
final_size = 10000
folder = '/eos/user/a/adpol/ceva'

file1 = h5py.File('%s/RSGraviton_bb_NARROW_%s-full.h5' % (folder, index), 'r')
file2 = h5py.File('%s/RSGraviton_hh_NARROW_%s-full.h5' % (folder, index), 'r')
file3 = h5py.File('%s/RSGraviton_tt_NARROW_%s-full.h5' % (folder, index), 'r')
file4 = h5py.File('%s/RSGraviton_WW_NARROW_%s-full.h5' % (folder, index), 'r')

save_path = '%s/fast/RSGraviton_NARROW_%s.h5' % (folder, index)

hdf5_dataset = h5py.File(save_path, 'w')

per_file_limit = int(final_size/4)

for column in ['labels', 'label_shapes', 'calorimeter']:
    print(column)
    if column == 'label_shapes':
        dt = h5py.special_dtype(vlen=np.uint8)
    else:
        dt = h5py.special_dtype(vlen=np.uint16)

    filtered_dataset = hdf5_dataset.create_dataset(name=column,
                                                   shape=(final_size,),
                                                   maxshape=(None,),
                                                   dtype=dt)

    for x, f in enumerate([file1, file2, file3, file4]):
        print(x)
        values = f.get(column).value[:per_file_limit]

        if column == 'calorimeter':
            values = np.vstack(values)

        if column == 'label_shapes':
            values = values.reshape(-1, 2)

        filtered_dataset[x*per_file_limit:(x+1)*per_file_limit] = values

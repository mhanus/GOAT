import os
import numpy

__author__ = 'Milan'

xs_names = \
{
  'St': 'St',
  'Ss': 'Ss',
  'nSf': 'nSf',
  'chi': 'chi',
  'Q': 'Q'
}

def parse_material(data_file_name):
  xs = ''
  new_xs = ''
  xs_data = dict()
  data_buffer = []

  with open(data_file_name, 'r') as f:
    for line in f:
      # Skip comments
      if line.startswith('*'):
        continue

      try:
        new_xs = xs_names[line.strip()]
      except KeyError:
        pass

      # Skip any header before the first xs
      if not new_xs:
        continue

      # If new xs has been encountered, save and reset the data buffer corresponding to the previous one (if not run
      # for the first time, in which case no buffer exists yet)
      if new_xs != xs:
        if data_buffer:
          # Transform the buffer for output
          data_array = numpy.atleast_1d(numpy.array(data_buffer).squeeze())

          # Assert consistency of xs data (or set the number of groups for the first time)
          try:
            assert data_array.shape[0] == G
          except NameError:
            G = data_array.shape[0]

          if data_array.size != G:

            try:
              # Scattering
              data_array = numpy.atleast_3d(data_array.reshape((-1,G,G)))
            except ValueError:
              try:
                # Source
                data_array = numpy.atleast_2d(data_array.reshape((-1,G)))
              except ValueError:
                #TODO: Invalid data format error
                raise
            else:
              # Assert consistency of xs data (or set the scattering order for the first time)
              try:
                assert data_array.shape[0] == K
              except NameError:
                K = data_array.shape[0]

          # Save the transformed buffer to the output dictionary
          xs_data[xs] = data_array

          # Reset the data buffer
          data_buffer = []

        xs = new_xs

      # look for new data
      data = line.replace(',', ' ').replace(';', ' ').split()
      if len(data) == 0:
        continue

      try:
        data_buffer.append(map(float, data))
      except ValueError:
        #TODO: Unexpected data error
        raise

  # If scattering is present, add the auxiliary pseudo-xs needed during assembling
  # TODO: Correct multigroup support (currently inter-group scattering is neglected
  try:
    St = xs_data['St']
    Ss = xs_data['Ss']
  except KeyError:
    # Set scattering order to 0 to indicate no scattering
    K = 0
  else:
    xs_data['C'] = numpy.zeros_like(Ss)
    for k in range(K):
      Ssk = numpy.diag(xs_data['Ss'][k])
      xs_data['C'][k,:,:] = numpy.diag( (2 * k + 1) * Ssk / (4 * numpy.pi * St * (St - Ssk)) )

  # Save the dictionary of xs data
  numpy.savez(os.path.splitext(data_file_name)[0] + '.npz', **xs_data)

  return G, K

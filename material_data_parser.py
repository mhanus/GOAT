import os
import numpy

__author__ = 'Milan'

class NewXS(Exception):
  pass

# noinspection PyUnboundLocalVariable
def parse_material(data_file_name):
  xs = ''
  new_xs = ''
  xs_data = dict()
  data_buffer = []

  xs_names = \
    {
      'St': 'St',
      'Ss': 'Ss',
      'nSf': 'nSf',
      'chi': 'chi',
      'Q': 'Q'
    }

  with open(data_file_name, 'r') as f:
    lines = f.readlines()

  for li, line in enumerate(lines):
    # Skip comments and empty lines
    if line.startswith(('*','\n')):
      continue

    try:
      new_xs = xs_names[line.strip()]
    except KeyError:
      # look for new data
      data = line.replace(',', ' ').replace(';', ' ').split()

      try:
        data_buffer.append(map(float, data))
      except ValueError:
        # TODO: Unexpected data error
        raise

    if new_xs != xs or li == len(lines)-1:
      # If new xs has been encountered, save and reset the data buffer corresponding to the previous one (if not run
      # for the first time, in which case no buffer exists yet)
      if data_buffer:
        # Transform the buffer for output
        data_array = numpy.atleast_1d(numpy.array(data_buffer).squeeze())

        # Assert consistency of xs data (or set the number of groups for the first time)
        try:
          assert data_array.shape[0] == G
        except NameError:
          G = data_array.shape[0]

        if xs == 'Ss':
          data_array = numpy.atleast_3d(data_array.reshape((-1,G,G)))
          K = data_array.shape[0]
        elif xs == 'Q':
          data_array = numpy.atleast_2d(data_array.reshape((-1,G)))

        # Save the transformed buffer to the output dictionary
        xs_data[xs] = data_array

        # Reset the data buffer
        data_buffer = []

      xs = new_xs

  St = xs_data['St']
  xs_data['D'] = 1/St

  # If scattering is present, add the auxiliary pseudo-xs needed during assembling
  # TODO: Correct multigroup support (currently inter-group scattering is neglected
  try:
    Ss = xs_data['Ss']
  except KeyError:
    # Set scattering order to 0 to indicate no scattering
    K = 0
  else:
    xs_data['C'] = numpy.zeros_like(Ss)
    for k in range(K):
      Ssk = numpy.diag(xs_data['Ss'][k])
      xs_data['C'][k,:,:] = numpy.diag( (2 * k + 1) * Ssk / (4 * numpy.pi * St * (St - Ssk)) )

  xs_data['dimensions'] = (G, K)

  # Save the dictionary of xs data
  numpy.savez(os.path.splitext(data_file_name)[0] + '.npz', **xs_data)

  return G, K
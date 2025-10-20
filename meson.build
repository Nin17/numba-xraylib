project(
  'numba_xraylib',
  'c',
  version: '0.2.0',
  default_options: [
    'buildtype=release',
  ]
  )

subproject('xraylib',
    default_options: [
        'default_library=shared',
        'python-bindings=disabled',
        'python-numpy-bindings=disabled',
        'fortran-bindings=disabled',
    ],
    required: true,
)

py3 = import('python').find_installation(pure: false)

subdir('numba_xraylib')

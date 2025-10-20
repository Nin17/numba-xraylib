
python_sources = [
  '__init__.py',
  'config.py',
  'config.toml',
  'overload_xraylib.py',
  'overload_xraylib_np.py',
  'utils.py',
]

xraylib_proj = subproject('xraylib',
    default_options: [
        'default_library=shared',
        'python-bindings=disabled',
        'python-numpy-bindings=disabled',
        'fortran-bindings=disabled',
    ],
    required: false,
)

py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'numba_xraylib'
)
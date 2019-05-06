const { _package, banner } = require('./helper');
const baseConfig = require('./rollup-base-config');

module.exports = Object.assign(baseConfig, {
  output: [
    {
      file: _package.main,
      format: 'umd',
      name: _package.namespace,
      banner: banner,
    },
    {
      file: _package.commonjs,
      format: 'cjs',
      banner: banner,
    },
    {
      file: _package.module,
      format: 'es',
      banner: banner,
    }
  ]
});

const serve = require('rollup-plugin-serve');
const typescript = require('rollup-plugin-typescript2');
const multiEntry = require('rollup-plugin-multi-entry');
const { banner, resolve } = require('./helper');
const baseConfig = require('./rollup-base-config');

baseConfig.plugins.push(// Default options
  serve({
    open: true,
    contentBase: [
      'examples', 'dist',
    ],
    host: 'localhost',
    port: 3003,
    headers: {
      'Access-Control-Allow-Origin': '*',
    }
  })
);

const config = Object.assign(baseConfig, {
  input: [
    resolve('examples/index.ts'),
  ],
  output: {
    file: './examples/index.js',
    format: 'iife',
    // name: _package.namespace,
    banner: banner,
  },
  external: undefined
});

const index = config.plugins.findIndex(item => item.name === 'rpt2');
const index1 = config.plugins.findIndex(item => item.name === 'tslint');
config.plugins.splice(index, 1, typescript({
  tsconfig: 'tsconfig-demo.json',
  clean: false,
  // outDir: resolve('types/'),
  declarationDir: 'examples',
  useTsconfigDeclarationDir: false,
}),);

config.plugins.splice(index1, 1);

// Use multiple entry points in your rollup bundle.
config.plugins.push(multiEntry({
  exports: false
}));

module.exports = config;

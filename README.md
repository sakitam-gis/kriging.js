kriging.js

[![Build Status](https://www.travis-ci.org/sakitam-gis/kriging.js.svg?branch=master)](https://www.travis-ci.org/sakitam-gis/kriging.js)
[![codecov](https://codecov.io/gh/sakitam-gis/kriging.js/branch/master/graph/badge.svg)](https://codecov.io/gh/sakitam-gis/kriging.js)
[![NPM downloads](https://img.shields.io/npm/dm/@sakitam-gis/kriging.svg)](https://npmjs.org/package/@sakitam-gis/kriging)
![JS gzip size](http://img.badgesize.io/https://unpkg.com/@sakitam-gis/kriging/dist/kriging.js?compression=gzip&label=gzip%20size:%20JS)
[![Npm package](https://img.shields.io/npm/v/@sakitam-gis/kriging.svg)](https://www.npmjs.org/package/@sakitam-gis/kriging)
[![GitHub stars](https://img.shields.io/github/stars/sakitam-gis/kriging.js.svg)](https://github.com/sakitam-gis/kriging.js/stargazers)
[![GitHub license](https://img.shields.io/github/license/sakitam-gis/kriging.js.svg)](https://github.com/sakitam-gis/kriging.js/blob/master/LICENSE)

## Dev

```bash
git clone https://github.com/sakitam-gis/kriging.js
npm install or yarn
npm run dev
npm run build
```

## Use

### CDN

```bash
https://unpkg.com/@sakitam-gis/kriging/dist/kriging.min.js
https://unpkg.com/@sakitam-gis/kriging/dist/kriging.js
```

### PACKAGES

```bash
npm i @sakitam-gis/kriging

# node
const kriging = require('@sakitam-gis/kriging');

# es
import kriging from '@sakitam-gis/kriging';
# or
import { train, grid } from '@sakitam-gis/kriging';

```

**kriging.js** is a Javascript library providing spatial prediction and mapping capabilities via the ordinary kriging algorithm. 

Kriging is a type of gaussian process where 2-dimensional coordinates are mapped to some target variable using kernel regression. This algorithm has been specifically designed to accurately model smaller data sets by assigning a prior to the variogram parameters.

Fitting a Model
---------------

The first step is to link **kriging.js** to your html code and assign your coordinate and target variables to 3 separate arrays.

``` html
<script src="kriging.js" type="text/javascript"></script>
<script type="text/javascript">
	var t = [ /* Target variable */ ];
	var x = [ /* X-axis coordinates */ ];
	var y = [ /* Y-axis coordinates */ ];
	var model = "exponential";
	var sigma2 = 0, alpha = 100;
	var variogram = kriging.train(t, x, y, model, sigma2, alpha);
</script>
```

The train method in the kriging object fits your input to whatever variogram model you specify - gaussian, exponential or spherical - and returns a variogram object. 

Error and Bayesian Prior
------------------------

Notice the σ<sup>2</sup> (sigma2) and α (alpha) variables, these correspond to the variance parameters of the gaussian process and the prior of the variogram model, respectively. A diffuse α prior is typically used; a formal mathematical definition of the model is provided below.

Predicting New Values
---------------------

Values can be predicted for new coordinate pairs by using the predict method in the kriging object.

``` javascript
  var xnew, ynew /* Pair of new coordinates to predict */;
  var tpredicted = kriging.predict(xnew, ynew, variogram);
  
```

Creating a Map
--------------

Variogram and Probability Model
-------------------------------

The various variogram models can be interpreted as kernel functions for 2-dimensional coordinates **a**, **b** and parameters nugget, range, sill and A. Reparameterized as a linear function, with w = [nugget, (sill-nugget)/range], this becomes:
  
- Gaussian: k(**a**,**b**) = w[0] + w[1] * ( 1 - exp{ -( ||**a**-**b**|| / range )<sup>2</sup> / A } )
- Exponential: k(**a**,**b**) = w[0] + w[1] * ( 1 - exp{ -( ||**a**-**b**|| / range ) / A } )
- Spherical: k(**a**,**b**) = w[0] + w[1] * ( 1.5 * ( ||**a**-**b**|| / range ) - 0.5 * ( ||**a**-**b**|| / range )<sup>3</sup> )

The variance parameter α of the prior distribution for w should be manually set, according to:

- w ~ N(w|**0**, α**I**)

Using the fitted kernel function hyperparameters and setting K as the Gram matrix, the prior and likelihood for the gaussian process become:

- **y**       ~ N(**y**|**0**, **K**)
- **t**|**y** ~ N(**t**|**y**, σ<sup>2</sup>**I**)

The variance parameter σ<sup>2</sup> of the likelihood reflects the error in the gaussian process and should be manually set. 





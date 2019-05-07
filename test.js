const fs = require('fs-extra');
const kriging = require('.');

// With Promises:
fs.readJson('./examples/csvjson.json')
  .then(res => {
    // console.log(res);
    const lngs = [];
    const lats = [];
    const values = [];
    for (let i = 0; i < res.length; i++) {
      const temp = res[i].temp;
      const x = res[i].lon;
      const y = res[i].lat;
      if (temp !== 999999 && i < 500) {
        const x1 = Number(x);
        const y1 = Number(y);
        const z1 = Number(temp);
        if (x1 && y1 && z1 !== undefined && !isNaN(z1)) {
          lngs.push(x1);
          lats.push(y1);
          values.push(z1);
        }
      }
    }

    console.time('train');
    const variogram = kriging.train(values, lngs, lats, 'exponential', 0, 100);
    console.timeEnd('train');
    // console.log(variogram);

    const mathGrid = true;

    if (mathGrid) {
      const extent = [
        63.81489, 12.770034,
        143.53648, 56.38334,
      ];
      console.time('grid');
      const vgrid = kriging.grid([
        [
          [extent[0], extent[1]], [extent[0], extent[3]],
          [extent[2], extent[3]], [extent[2], extent[1]],
        ],
      ], variogram, 0.3986079500000001);
      console.timeEnd('grid');
      // console.log(vgrid);
    }
  })
  .catch(err => {
    console.error(err)
  });

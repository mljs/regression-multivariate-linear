import MLR from '..';

var fs = require('fs');
var path = require('path');

var Papa = require('papaparse');

// Dataset -> https://openmv.net/info/distillation-tower

describe('multivariate linear regression', () => {
  it('should work with large dataset with intercept', () => {
    const ymatrix = fs.readFileSync(path.resolve(path.join('./data/ydistillationjs.csv')), 'utf8');
    const y = Papa.parse(ymatrix, { delimiter: ',' }).data.map((a) => a.map((b) => parseFloat(b)));
    const xmatrix = fs.readFileSync(path.resolve(path.join('./data/xdistillationjs.csv')), 'utf8');
    const x = Papa.parse(xmatrix, { delimiter: ',' }).data.map((a) => a.map((b) => parseFloat(b)));
    //
    const mlr = new MLR(x, y, { intercept: true });
    expect(mlr.predict(x[0]).map(Math.round)).toStrictEqual([33]);
    expect(mlr.predict(x[10]).map(Math.round)).toStrictEqual([35]);
    expect(mlr.predict(x[20]).map(Math.round)).toStrictEqual([41]);
    expect(mlr.predict(x[30]).map(Math.round)).toStrictEqual([33]);
    expect(mlr.predict(x[40]).map(Math.round)).toStrictEqual([46]);
    expect(mlr.predict(x[50]).map(Math.round)).toStrictEqual([33]);
    expect(mlr.predict(x[60]).map(Math.round)).toStrictEqual([35]);
    expect(mlr.predict(x[250]).map(Math.round)).toStrictEqual([36]);
  });
  it('should work with large dataset without intercept', () => {
    const ymatrix = fs.readFileSync(path.resolve(path.join('./data/ydistillationjs.csv')), 'utf8');
    const y = Papa.parse(ymatrix, { delimiter: ',' }).data.map((a) => a.map((b) => parseFloat(b)));
    const xmatrix = fs.readFileSync(path.resolve(path.join('./data/xdistillationjs.csv')), 'utf8');
    const x = Papa.parse(xmatrix, { delimiter: ',' }).data.map((a) => a.map((b) => parseFloat(b)));
    //
    var mlr = new MLR(x, y, { intercept: false });
    expect(mlr.predict(x[0]).map(Math.round)).toStrictEqual([33]);
    expect(mlr.predict(x[10]).map(Math.round)).toStrictEqual([34]);
    expect(mlr.predict(x[20]).map(Math.round)).toStrictEqual([41]);
    expect(mlr.predict(x[30]).map(Math.round)).toStrictEqual([33]);
    expect(mlr.predict(x[40]).map(Math.round)).toStrictEqual([45]);
    expect(mlr.predict(x[50]).map(Math.round)).toStrictEqual([33]);
    expect(mlr.predict(x[60]).map(Math.round)).toStrictEqual([35]);
    expect(mlr.predict(x[250]).map(Math.round)).toStrictEqual([36]);
  });
});

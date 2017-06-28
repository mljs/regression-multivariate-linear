import MLR from '..';

describe('multivariate linear regression', () => {
    it('should work with 2 inputs and 3 outputs', () => {
        const mlr = new MLR(
            [[0, 0], [1, 2], [2, 3], [3, 4]],
            // y0 = 2 * x0, y1 = 2 * x1, y2 = x0 + x1
            [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]]
        );
        expect(mlr.predict([2, 3]).map(Math.round)).toEqual([4, 6, 5]);
        expect(mlr.predict([4, 4]).map(Math.round)).toEqual([8, 8, 8]);
    });

    it('should work with 2 inputs and 3 outputs - intercept is 0', () => {
        const mlr = new MLR(
            [[0, 0], [1, 2], [2, 3], [3, 4]],
            // y0 = 2 * x0, y1 = 2 * x1, y2 = x0 + x1
            [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]],
            {intercept: true}
        );
        expect(mlr.predict([2, 3]).map(Math.round)).toEqual([4, 6, 5]);
        expect(mlr.predict([4, 4]).map(Math.round)).toEqual([8, 8, 8]);
    });

    it('should work with 2 inputs and 3 outputs - intercept is not 0', () => {
        const mlr = new MLR(
            [[0, 0], [1, 2], [2, 3], [3, 4]],
            // y0 = 2 * x0 -1, y1 = 2 * x1 + 2, y2 = x0 + x1 + 10
            [[-1, 2, 10], [1, 6, 13], [3, 8, 15], [5, 10, 17]],
            {intercept: true}
        );
        expect(mlr.predict([2, 3]).map(Math.round)).toEqual([3, 8, 15]);
        expect(mlr.predict([4, 4]).map(Math.round)).toEqual([7, 10, 18]);
    });

    it('should work with 2 inputs and 1 output (x02)', () => {
        const data = require('../../data/x02');
        const mlr = new MLR(
            data.x,
            data.y,
            {intercept: false}
        );
        const prediction = mlr.predict(data.x);
        expect(prediction[0][0]).toBeCloseTo(38.05);
    });

    it('should work with 2 inputs and 1 output (x42)', () => {
        const data = require('../../data/x42');
        const mlr = new MLR(
            data.x,
            data.y,
            {intercept: false}
        );
        const expectedWeights = [83.125, 2.625, 3.125, 3.75, -2.0, -4.375, 0.0, 1.5, -0.25];
        for (let i = 0; i < mlr.weights.length; i++) {
            expect(mlr.weights[i][0]).toBeCloseTo(expectedWeights[i]);
        }
    });

    it('toJSON and load', () => {
        const mlr = new MLR(
            [[0, 0], [1, 2], [2, 3], [3, 4]],
            [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]]
        );
        const json = JSON.parse(JSON.stringify(mlr));
        const newMlr = MLR.load(json);
        expect(newMlr.predict([2, 3]).map(Math.round)).toEqual([4, 6, 5]);
        expect(newMlr.predict([4, 4]).map(Math.round)).toEqual([8, 8, 8]);
    });

    it('datamining test 1-1', () => {
        const X = [[4.47], [208.30], [3400.00]];
        const Y = [[0.51], [105.66], [1800.00]];
        const mlr = new MLR(X, Y, {intercept: true});
        expect(mlr.weights[0][0]).toBeCloseTo(0.53);
        expect(mlr.weights[1][0]).toBeCloseTo(-3.29);
    });

    it('datamining test 1-2', () => {
        const X = [[4.47, 1], [208.30, 1], [3400.00, 1]];
        const Y = [[0.51], [105.66], [1800.00]];
        const mlr = new MLR(X, Y, {intercept: false});
        expect(mlr.weights[0][0]).toBeCloseTo(0.53);
        expect(mlr.weights[1][0]).toBeCloseTo(-3.29);
    });

    it('datamining test 2', () => {
        const X = [[1, 1, 1], [2, 1, 1], [3, 1, 1]];
        const Y = [[2, 3], [4, 6], [6, 9]];
        const mlr = new MLR(X, Y);
        expect(mlr.weights[0][0]).toBeCloseTo(2);
        expect(mlr.weights[0][1]).toBeCloseTo(3);
        expect(mlr.weights[1][0]).toBeCloseTo(0);
        expect(mlr.weights[1][1]).toBeCloseTo(0);
        expect(mlr.weights[2][0]).toBeCloseTo(0);
        expect(mlr.weights[2][1]).toBeCloseTo(0);
    });
});

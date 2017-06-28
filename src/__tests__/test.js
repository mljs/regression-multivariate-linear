import MLR from '..';

describe('multivariate linear regression', () => {
    it('should work with 2 inputs and 3 outputs', () => {
        const mlr = new MLR(
            [[0, 0], [1, 2], [2, 3], [3, 4]],
            [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]]
        );
        expect(mlr.predict([2, 3]).map(Math.round)).toEqual([4, 6, 5]);
        expect(mlr.predict([4, 4]).map(Math.round)).toEqual([8, 8, 8]);
    });

    it('should work with 2 inputs and 1 output (x02)', () => {
        const data = require('../../data/x02');
        const mlr = new MLR(
            data.x,
            data.y
        );
        const prediction = mlr.predict(data.x);
        expect(prediction[0][0]).toBeCloseTo(38.05);
    });

    it('should work with 2 inputs and 1 output (x42)', () => {
        const data = require('../../data/x42');
        const mlr = new MLR(
            data.x,
            data.y
        );
        const expectedWeights = [83.125, 2.625, 3.125, 3.75, -2.0, -4.375, 0.0, 1.5, -0.25];
        for (let i = 0; i < mlr.weights.length; i++) {
            expect(mlr.weights[i][0]).toBeCloseTo(expectedWeights[i]);
        }
    });
});

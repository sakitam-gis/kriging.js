import kriging from '../src';

describe('indexSpec', () => {
  beforeEach(() => {
  });

  afterEach(() => {
  });

  describe('utils', () => {
    it('utils.min', () => {
      const min = kriging.min([1, 2, 3.5, 0.2]);
      expect(min).toBe(0.2);
    });

    it('utils.max', () => {
      const max = kriging.max([1, 2, 3.5, 0.2]);
      expect(max).toBe(3.5);
    });
  });
});

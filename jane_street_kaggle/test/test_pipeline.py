import pkg_resources
import pytest


@pytest.fixture
def data():
    import pandas as pd
    res_stream = pkg_resources.resource_stream(__name__, 'resources/float_ds.csv')
    return pd.read_csv(res_stream)


@pytest.fixture
def data_dask():
    import dask.dataframe as dd
    res_stream = pkg_resources.resource_filename(__name__, 'resources/float_ds.csv')
    return dd.read_csv(res_stream)


def _get_modules():
    from jane_street_kaggle.base import PipelineModule
    from jane_street_kaggle import modules
    pipeline_modules = [v for k, v in modules.__dict__.items() if isinstance(v, PipelineModule)]
    return pipeline_modules


def test(data, tmpdir: str):
    from jane_street_kaggle import modules
    from jane_street_kaggle.pipeline import run

    pipeline = [
        ('scale', modules.Scale()),
        ('impute', modules.Impute()),
        ('model', modules.Model()),
        ('action', modules.Action())
    ]
    run(data, pipeline, True, cache_dir=tmpdir)


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            module,
            id=module.__name__
        )
        for module in _get_modules()
    ],
)
def test_fit_immutable_dask(data_dask, module, tmpdir):
    from jane_street_kaggle.pipeline import run
    from pandas.testing import assert_frame_equal

    generic_pipeline = [
        (module.__name__, module()),
    ]

    data_before = data_dask.copy()
    run(data_dask, generic_pipeline, fit=True, cache_dir=tmpdir)
    assert_frame_equal(data_before.compute(), data_dask.compute())


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            module,
            id=module.__name__
        )
        for module in _get_modules()
    ],
)
def test_fit_immutable(data, module, tmpdir):
    from jane_street_kaggle.pipeline import run
    from pandas.testing import assert_frame_equal

    generic_pipeline = [
        (module.__name__, module()),
    ]

    data_before = data.copy()
    run(data, generic_pipeline, fit=True, cache_dir=tmpdir)
    assert_frame_equal(data_before, data)


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            module,
            id=module.__name__
        )
        for module in _get_modules()
    ],
)
def test_transform_immutable(data, module, tmpdir):
    from jane_street_kaggle.pipeline import run
    from pandas.testing import assert_frame_equal

    generic_pipeline = [
        (module.__name__, module()),
    ]

    run(data, generic_pipeline, fit=True, cache_dir=tmpdir)

    data_before = data.copy()
    data_transformed = run(data, generic_pipeline, fit=False, cache_dir=tmpdir)

    assert data_transformed is not data
    assert_frame_equal(data_before, data)


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            module,
            id=module.__name__
        )
        for module in _get_modules()
    ],
)
def test_transform_immutable_dask(data_dask, module, tmpdir):
    from jane_street_kaggle.pipeline import run
    from pandas.testing import assert_frame_equal

    generic_pipeline = [
        (module.__name__, module()),
    ]

    run(data_dask.compute(), generic_pipeline, fit=True, cache_dir=tmpdir)

    data_before = data_dask.compute().copy()
    data_transformed = run(data_dask, generic_pipeline, fit=False, cache_dir=tmpdir)

    assert data_transformed is not data
    assert_frame_equal(data_before, data_dask.compute())


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            module,
            id=module.__name__
        )
        for module in _get_modules()
    ],
)
def test_dask_pandas_equivalence(tmpdir, data, data_dask, module):
    from jane_street_kaggle.pipeline import run
    from pandas.testing import assert_frame_equal

    test_pipeline = [
        (module.__name__, module()),
    ]

    if module.__name__ == 'Scale':
        v_before = data_dask['feature_1'].std().compute()

    # Fit on pa
    run(data, test_pipeline, fit=True, cache_dir=tmpdir)
    pd_in__pd_out = run(data, test_pipeline, fit=False, cache_dir=tmpdir)
    pd_in__dask_out = run(data_dask, test_pipeline, fit=False, cache_dir=tmpdir).compute()

    if module.__name__ == 'Scale':
        v_after = data_dask['feature_1'].std().compute()

        assert v_after == v_before

    # Re-initialize pipeline
    test_pipeline = [
        (module.__name__, module()),
    ]

    # Fit on pa
    run(data_dask, test_pipeline, fit=True, cache_dir=tmpdir)
    dask_in__pd_out = run(data, test_pipeline, fit=False, cache_dir=tmpdir)
    dask_in__dask_out = run(data_dask, test_pipeline, fit=False, cache_dir=tmpdir).compute()

    assert_frame_equal(pd_in__pd_out, pd_in__dask_out)
    assert_frame_equal(pd_in__pd_out, dask_in__pd_out)
    assert_frame_equal(pd_in__pd_out, dask_in__dask_out)
    assert_frame_equal(pd_in__dask_out, dask_in__pd_out)
    assert_frame_equal(pd_in__dask_out, dask_in__pd_out)
    assert_frame_equal(dask_in__pd_out, dask_in__dask_out)

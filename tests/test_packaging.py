from nonos import InitParamNonos, FieldNonos

def test_packaging():
    # this will fail unless the package is installed properly
    pconfig = InitParamNonos(info=True).config
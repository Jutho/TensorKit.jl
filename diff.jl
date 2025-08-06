using Zygote
using TensorOperations

function expectation_value_1x1(O,
                               E_north, E_east, E_south, E_west,
                               C_northeast, C_northwest, C_southeast, C_southwest)
    E_north *= C_northeast
    E_east *= C_southeast
    E_south *= C_southwest
    E_west *= C_northwest

    @tensor ρ[dt; db] := E_south[χSE DSt DSb; χSW] * E_west[χSW DWt DWb; χNW] *
                         A[dt; DNt DEt DSt DWt] * E_north[χNW DNt DNb; χNE] *
                         E_east[χNE DEt DEb; χSE] * conj(Ā[db; DNb DEb DSb DWb])

    return tr(O * ρ)
end

E = expectation_value_1x1(O, edges..., corners...)
∂E_∂O, ∂E_∂edges..., ∂E_∂corners... = gradient(expectation_value_1x1, O, edges...,
                                               corners...)

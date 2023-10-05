# This file is largely inspired by the Base Julia file test/choostests.jl

const TESTNAMES = ["sectors", "spaces", "fusiontrees", "tensors", "planar", "ad"]
const TESTGROUPS = String[]

const SECTORNAMES = ["Trivial", "Z2Irrep", "Z3Irrep", "Z4Irrep", "U1Irrep", "CU1Irrep",
                     "SU2Irrep", "NewSU2Irrep", "FibonacciAnyon", "IsingAnyon",
                     "FermionParity", "FermionNumber", "FermionSpin", "Z3Irrep ⊠ Z4Irrep",
                     "FermionNumber ⊠ SU2Irrep", "FermionSpin ⊠ SU2Irrep",
                     "NewSU2Irrep ⊠ NewSU2Irrep", "FibonacciAnyon", "Object{E6}",
                     "Z2Irrep ⊠ FibonacciAnyon ⊠ FibonacciAnyon"]
const DEFAULT_SECTORNAMES = try
    if ENV["CI"] == "true"
        println("Detected CI environment")
        if Sys.iswindows()
            ["Trivial", "Z2Irrep", "FermionParity", "Z3Irrep", "U1Irrep",
             "FermionNumber", "CU1Irrep", "SU2Irrep"]
        elseif Sys.isapple()
            ["Trivial", "Z2Irrep", "FermionParity", "Z3Irrep", "FermionNumber",
             "FermionSpin"]
        else
            ["Trivial", "Z2Irrep", "FermionParity", "U1Irrep", "CU1Irrep", "SU2Irrep",
             "FermionSpin"]
        end
    else
        SECTORNAMES
    end
catch
    SECTORNAMES
end
const SECTORGROUPS = String[]

""" 
    (; tests, exit_on_error, seed) = choosetests(choices=[])

Selects a set of tests to be run. `choices` should be a vector of test names and/or
sectornames; if empty or set to `["all"]`, all tests are selected.

Several options can be passed to `choosetests` by including a special token in the `choices`
argument:
- "--skip", which makes all tests coming after be skipped
- "--exit-on-error" which sets the value of `exit_on_error`
- "--revise" which loads Revise
- "--seed=SEED", which sets the value of `seed` to `SEED` (parsed as an `UInt128`);
  `seed` is otherwise initialized randomly. This option can be used to reproduce failed tests.
- "--help", which prints a help message and then skips all tests.
- "--help-list", which prints the options computed without running them.

The function returns a named tuple with the following elements:
- `tests` is a vector of fully-expanded test names
- `sectors` is a vector of fully-expanded sector names
- `exit_on_error` is true if an error in one test should cancel remaining tests to be run
  (otherwise, all tests are run unconditionally)
- `seed` is a seed which will be used to initialize the global RNG for each test to be run.
"""
function choosetests(choices=[])
    tests = []
    sectors = []
    skip_tests = Set()
    skip_sectors = Set()
    exit_on_error = false
    use_revise = false
    seed = rand(RandomDevice(), UInt128)
    ci_option_passed = false
    dryrun = false

    # parse options
    for (i, t) in enumerate(choices)
        if t == "--skip"
            union!(skip_tests, choices[(i + 1):end])
            break
        elseif t == "--exit-on-error"
            exit_on_error = true
        elseif t == "--revise"
            use_revise = true
        elseif startswith(t, "--seed=")
            seed = parse(UInt128, t[(length("--seed=") + 1):end])
        elseif t == "--ci"
            ci_option_passed = true
        elseif t == "--help-list"
            dryrun = true
        elseif t == "--help"
            println("""
                USAGE: ./julia runtests.jl [options] [tests]
                OPTIONS:
                  --exit-on-error      : stop tests immediately when a test group fails
                  --help               : prints this help message
                  --help-list          : prints the options computed without running them
                  --revise             : load Revise
                  --seed=<SEED>        : set the initial seed for all testgroups (parsed as a UInt128)
                  --skip <NAMES>...    : skip test or collection tagged with <NAMES>
                TESTS:
                  Can be testsets, such as ($TESTNAMES),
                  or sector names, such as ($SECTORNAMES).

                  Prefixing a name with `-` (such as `-sectors`) can be used to skip a particular test.
                """)
            return (; tests=[],
                    sectors=[],
                    exit_on_error=false,
                    use_revise=false,
                    seed=UInt128(0))
        elseif startswith(t, "--")
            error("unknown option: $t")
        elseif startswith(t, "-")
            if t[2:end] in TESTNAMES || t in TESTGROUPS
                push!(skip_tests, t[2:end])
            elseif t[2:end] in SECTORNAMES || t in SECTORGROUPS
                push!(skip_sectors, t[2:end])
            else
                error("unknown test or sector name at $i: $t")
            end
        else
            if t in TESTNAMES || t in TESTGROUPS
                push!(tests, t)
            elseif t in SECTORNAMES || t in SECTORGROUPS
                push!(sectors, t)
            elseif t == "all-tests"
                append!(tests, TESTNAMES)
            elseif t == "all-sectors"
                append!(sectors, SECTORNAMES)
            elseif t == "default-sectors"
                append!(sectors, DEFAULT_SECTORNAMES)
            else
                error("unknown test or sector name at $i: $t")
            end
        end
    end

    # Filter tests
    unhandled_tests = copy(skip_tests)

    if isempty(tests)
        append!(tests, TESTNAMES)
    end

    # Functionality to add testgroups for inclusion/exclusion:
    # function filtertests!(tests, name_or_group, names=[name_or_group])
    #     flt = x -> (x != name_or_group && !(x in names))
    #     if name_or_group in skip_tests
    #         filter!(flt, tests)
    #         pop!(unhandled_tests, name_or_group)
    #     elseif name_or_group in tests
    #         filter!(flt, tests)
    #         prepend!(tests, names)
    #     end
    # end

    filter!(!in(tests), unhandled_tests)
    filter!(!in(skip_tests), tests)

    # Filter sectors
    unhandled_sectors = copy(skip_sectors)

    if isempty(sectors)
        append!(sectors, DEFAULT_SECTORNAMES)
    end

    # Functionality to add sectorgroups for inclusion/exclusion:
    # function filtersectors!(sectors, name_or_group, names=[name_or_group])
    #     flt = x -> (x != name_or_group && !(x in names))
    #     if name_or_group in skip_sectors
    #         filter!(flt, sectors)
    #         pop!(unhandled_sectors, name_or_group)
    #     elseif name_or_group in sectors
    #         filter!(flt, sectors)
    #         prepend!(sectors, names)
    #     end
    # end

    filter!(!in(sectors), unhandled_sectors)
    filter!(!in(skip_sectors), sectors)

    return (; tests, sectors, exit_on_error, use_revise, seed)
end

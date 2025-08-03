# Changelog

## [0.7.0](https://github.com/mkm29/stockula/compare/v0.6.1...v0.7.0) (2025-08-03)


### Features

* add backtest optimized allocator ([7fee370](https://github.com/mkm29/stockula/commit/7fee37083c0fa22afa2ebfe845c9d98b88aecc77))
* Add backtest-optimized configuration and update main logic for saving configurations ([5d4ab46](https://github.com/mkm29/stockula/commit/5d4ab465ea800ebdc4954b4ce94946b0ebf2653b))
* Implement backtest-optimized asset allocation strategy ([ba2fe24](https://github.com/mkm29/stockula/commit/ba2fe2476aa015710621b0bf4949d6bc4df3500c))
* Implement Backtest-Optimized Asset Allocation Strategy ([b08e60f](https://github.com/mkm29/stockula/commit/b08e60f3b89c71ec15be5744b79d4f69a90c5650))


### Bug Fixes

* Rename loop variable for clarity and remove unused mock strategy ([a0209d9](https://github.com/mkm29/stockula/commit/a0209d9a326ca9080da4500a67d2b3a0f65d6cba))

## [0.6.1](https://github.com/mkm29/stockula/compare/v0.6.0...v0.6.1) (2025-08-02)

### Bug Fixes

- resolved timezone discrepancies in backtesting ([2619b88](https://github.com/mkm29/stockula/commit/2619b8862423eabeaea1f560801062b06ce96ccf))
- resolved timezone discrepancies in backtesting ([9e95e95](https://github.com/mkm29/stockula/commit/9e95e95b2678f80da7b27aacd52f95dc4c9032ed))

## [0.6.0](https://github.com/mkm29/stockula/compare/v0.5.2...v0.6.0) (2025-08-02)

### Features

- create allocator class ([53cfde9](https://github.com/mkm29/stockula/commit/53cfde9d4fd524a7842fb7adae4b1120bfc49f60))

### Miscellaneous Chores

- update linting commands for consistency with CI and improve documentation ([d02ffa7](https://github.com/mkm29/stockula/commit/d02ffa7207109987fa0a6d7967fae60d61759f36))

## [0.5.2](https://github.com/mkm29/stockula/compare/v0.5.1...v0.5.2) (2025-08-02)

### Bug Fixes

- require successful run of test GitHub Action before docker-build is run ([d087013](https://github.com/mkm29/stockula/commit/d08701394061c9ff5c401bde3a90b73a3d3bca70))
- require successful run of test GitHub Action before docker-build… ([38e94a8](https://github.com/mkm29/stockula/commit/38e94a81a5b3afa5da6e04f891e036a6cac23585))

## [0.5.1](https://github.com/mkm29/stockula/compare/v0.5.0...v0.5.1) (2025-08-02)

### Bug Fixes

- removed unused packages from pyproject.toml ([cf2abac](https://github.com/mkm29/stockula/commit/cf2abacd83f6eec1a20f47c3cc3dad4a021b48a9))
- removed unused packages from pyproject.toml ([8bf22e1](https://github.com/mkm29/stockula/commit/8bf22e189b0b0167399c5f415464fa5918668884))

## [0.5.0](https://github.com/mkm29/stockula/compare/v0.4.3...v0.5.0) (2025-08-02)

### Features

- Enhance documentation and testing coverage ([35aa9e4](https://github.com/mkm29/stockula/commit/35aa9e4ad7ed0f70ed2c00b033bf3a0b5ff87fc4))

### Bug Fixes

- Enhance documentation and testing coverage ([3ffcc96](https://github.com/mkm29/stockula/commit/3ffcc967594a3f47aae83100bc9eb9dc89ae5a3e))
- improve test coverage ([f19dffe](https://github.com/mkm29/stockula/commit/f19dffe92aa584ff1260a7e91f46fde692dabf09))

### Code Refactoring

- clean up Hadolint config and enhance test coverage for configuration loading ([c25c2c3](https://github.com/mkm29/stockula/commit/c25c2c3c0c2599e1729b105889813922f2065999))

### Miscellaneous Chores

- Update test workflow to simplify unit test command and add JUnit XML reporting ([55aa3a1](https://github.com/mkm29/stockula/commit/55aa3a11e23d7b86d2136c9e10b109a3ed53bd7b))

## [0.4.3](https://github.com/mkm29/stockula/compare/v0.4.2...v0.4.3) (2025-08-01)

### Bug Fixes

- improve test coverage ([cfeafd1](https://github.com/mkm29/stockula/commit/cfeafd1caf2e2b78d1f90e7ed357149855b53410))

## [0.4.2](https://github.com/mkm29/stockula/compare/v0.4.1...v0.4.2) (2025-08-01)

### Bug Fixes

- changed the tagging of docker images in docker-build.yml ([71f17e6](https://github.com/mkm29/stockula/commit/71f17e600160646609aacfe0d715742eaa4ddcaf))
- reverted to previous docker build (only build cli) ([237ee63](https://github.com/mkm29/stockula/commit/237ee63ed6ba9b3cb5fc29a6b9352a99319517e6))

## [0.4.1](https://github.com/mkm29/stockula/compare/v0.4.0...v0.4.1) (2025-08-01)

### Bug Fixes

- added disk space cleanup acton ([1496631](https://github.com/mkm29/stockula/commit/1496631bb3e8d861e7fb84b5b1b39ce72c6c0b2f))
- added disk space cleanup acton ([0b5278b](https://github.com/mkm29/stockula/commit/0b5278bf820a4062c5f6cdaa139b13ce411f3a9c))

## [0.4.0](https://github.com/mkm29/stockula/compare/v0.3.2...v0.4.0) (2025-08-01)

### Features

- enhance Docker workflows and documentation for production and C… ([43c4523](https://github.com/mkm29/stockula/commit/43c4523a2870f62aa91300d1451a6b8887eccc02))
- enhance Docker workflows and documentation for production and CLI images ([d96038d](https://github.com/mkm29/stockula/commit/d96038d4909fb9ce7c816508d013ec5bdac84eb2))

## [0.3.2](https://github.com/mkm29/stockula/compare/v0.3.1...v0.3.2) (2025-08-01)

### Bug Fixes

- optimized Dockerfile and build workflow for running on GitHub Ac… ([f9f52ac](https://github.com/mkm29/stockula/commit/f9f52acb195a65dbbb8b3710491ae78cc1920ff9))
- optimized Dockerfile and build workflow for running on GitHub Actions ([13106b6](https://github.com/mkm29/stockula/commit/13106b6bd2c7428ef16b655ed515b3b1cde8cdc6))

## [0.3.1](https://github.com/mkm29/stockula/compare/v0.3.0...v0.3.1) (2025-08-01)

### Bug Fixes

- adjust versioning ([1db4132](https://github.com/mkm29/stockula/commit/1db41329608859e281c24c03923472598ec64971))

### Miscellaneous Chores

- downgrade release version to 0.2.0 in manifest ([0a69553](https://github.com/mkm29/stockula/commit/0a6955318686ea8675f3e0c95e308c3d8315ea8c))
- update Codecov action to version 5 in test workflow ([7e985ac](https://github.com/mkm29/stockula/commit/7e985ac1d1a892b1e9c425a05a29c7f491fcc209))

## [0.3.0](https://github.com/mkm29/stockula/compare/v0.2.1...v0.3.0) (2025-08-01)

### Features

- Add initial versioning configuration and changelog documentation ([b134945](https://github.com/mkm29/stockula/commit/b134945a74a48d472e7859ef9a5879ff332a89c8))
- add pytest-xdist for parallel test execution and improve test d… ([e07846a](https://github.com/mkm29/stockula/commit/e07846a4a5ece26ad576fa5d81e2119bf4d059c7))
- add pytest-xdist for parallel test execution and improve test database handling ([006a288](https://github.com/mkm29/stockula/commit/006a288ebe50754418395f720a6cec4c7a1e1cbb))
- added pre-commit hooks to lint code, check data types ([4e577fb](https://github.com/mkm29/stockula/commit/4e577fb442e7ed24c618f8f8117b905d0d84e887))
- **config:** Add release configuration and manifest for version 0.2.… ([08486a4](https://github.com/mkm29/stockula/commit/08486a40bc9cbf5e593a56aa6f8bee8e89afcd7f))
- **config:** Add release configuration and manifest for version 0.2.0; update .gitignore to exclude JSON files ([7a21aff](https://github.com/mkm29/stockula/commit/7a21affad0efb0e4347452821f09c494265c709a))
- **docs:** Add development documentation for AutoTS threading considerations ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- **docs:** Update changelog and user guide for train/test evaluation enhancements ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- **docs:** Update changelog and user guide for train/test evaluation… ([82c2fbb](https://github.com/mkm29/stockula/commit/82c2fbbce7eb0ae10863ed3348e5125cd2c3f60e))
- **docs:** Update changelog for version 0.2.0; enhance forecast evaluation and backtest functionality ([e061515](https://github.com/mkm29/stockula/commit/e061515b033024d0ee2c3ac9e609a035b00b09eb))
- **domain:** Exclude tickers with 0% allocation from portfolio creation and tests ([ac987dc](https://github.com/mkm29/stockula/commit/ac987dc3dccd2ae83c5b9f6552dbca0e92a4af8e))
- Enhance main.py with portfolio summary and initial value calculations ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- **forecasting:** Implement train/test split evaluation for stock price forecasting ([ce74bd6](https://github.com/mkm29/stockula/commit/ce74bd64b610ac2e6b8dc1407f2dcc8864a77e24))
- Implement Ticker domain model with singleton registry ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Initialize SQLite database for stock data storage ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- **logging:** Add method to set module-specific logging levels ([8b95356](https://github.com/mkm29/stockula/commit/8b953561b30c73f58bd0d270e7b887f688a762e0))
- **logging:** Add method to set module-specific logging levels ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **logging:** Update LoggingManager to handle additional third-party libraries ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **main:** Add detailed portfolio holdings display in console output ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))
- **main:** Enhance forecasting output with return percentage and improve logging ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **main:** Enhance portfolio value display with forecasted values and accuracy metrics ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- update CI/CD workflows and documentation ([0e48c3d](https://github.com/mkm29/stockula/commit/0e48c3d4c4c637800e318f95276970ff611b9e8f))

### Bug Fixes

- **backtesting:** Suppress progress output in backtest runner ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- Correct import paths in main.py ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Display correct strategy-specific backtest results in portfolio summary ([6bd5a97](https://github.com/mkm29/stockula/commit/6bd5a9786e6662448d73229c387fe2c38ed1b788))
- **docs:** Update documentation links to use relative paths ([a5079ba](https://github.com/mkm29/stockula/commit/a5079bad4b6aaec35a8b19b3d8918ae0dd5e3686))
- update .gitignore and CHANGELOG for CLAUDE file renaming ([324c7e6](https://github.com/mkm29/stockula/commit/324c7e615841c48c713a03085515b7da490d79d6))
- update .gitignore and CHANGELOG for CLAUDE file renaming ([4fd14b1](https://github.com/mkm29/stockula/commit/4fd14b1cf4dde2500690d22ddb91fd8af0edc907))
- update README links to ensure proper navigation ([bd5e1df](https://github.com/mkm29/stockula/commit/bd5e1dff819f38d109ba041a84cd06b055bbf001))
- update README links to ensure proper navigation ([6563c19](https://github.com/mkm29/stockula/commit/6563c1949e250b21b964b264d47252457dc83d50))

### Documentation

- Add broker configuration documentation to README ([e474458](https://github.com/mkm29/stockula/commit/e47445876c2437c3fe5cea99054d587e308ea36b))
- Update README to include system requirements, dependencies, architecture overview, and data flow ([505b797](https://github.com/mkm29/stockula/commit/505b7973878f7e6fd541e12c79f02b9b61074aea))
- Update stockula.yml and stockula.yml.example with new structure ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Update storage requirement in README for data caching ([da020fa](https://github.com/mkm29/stockula/commit/da020fab7edd3e1de4a962f2620f85b4cf26f2a5))
- Update user guide and README to reflect sorting by return in analysis modes ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))

### Code Refactoring

- Create structured data models for backtest results ([d8b5fe2](https://github.com/mkm29/stockula/commit/d8b5fe2ff3a1ce63265fdf8afb5132e83a16adcf))
- Organize configuration files into examples directory ([ecc4c0b](https://github.com/mkm29/stockula/commit/ecc4c0bd62b57f47cbbe4fc5887f341cd12006ae))
- Update forecaster.py to streamline imports ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))

### Tests

- **autots:** Create direct test for AutoTS to isolate hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add minimal test configuration for two tickers ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add quick test configuration with a single ticker ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add simple test configuration for forecast testing ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add test configuration for parallel progress tracking ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **parallel:** Implement tests for parallel forecasting to debug hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **progress:** Create test script to demonstrate parallel forecasting progress tracking ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **sequential:** Implement sequential forecasting test to verify functionality ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **single:** Create test for single ticker forecasting to debug hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **unit:** Refactor main tests to use parallel forecasting and improve logging checks ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **unit:** Update unit tests for forecaster to check output suppression ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))

### Miscellaneous Chores

- Add CLAUDE.market_data and .claude to .gitignore ([0cf0065](https://github.com/mkm29/stockula/commit/0cf006507c51b49bbd8b66b165c33e28434ccb81))
- Add example configuration files for different allocation strategies ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Add SQLite database initialization and testing script ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- comment out pull_request branch triggers in workflow files ([bc9ee6f](https://github.com/mkm29/stockula/commit/bc9ee6f54cc4c1ccdcd6826345b73cafdfa6a77f))
- copy in README.md in Dockerfile ([f6a8db0](https://github.com/mkm29/stockula/commit/f6a8db0f69652d56e8e562db6c44d8ae266a93cf))
- fixed pip install command in Dockerfile ([4765432](https://github.com/mkm29/stockula/commit/4765432a8cde9a789cbb16b8ea72fd2a42d408fc))
- **main:** release 0.2.0 ([dd76ab5](https://github.com/mkm29/stockula/commit/dd76ab5a102148ac6c6c2d80481b03ec348a1786))
- **main:** release 0.2.0 ([5fe8f0e](https://github.com/mkm29/stockula/commit/5fe8f0e9ddab552546f5843e8bd6695c782799d0))
- **main:** release stockula 0.2.0 ([e779ca3](https://github.com/mkm29/stockula/commit/e779ca31ad6122c3320b419ce4766f851053e6c8))
- **main:** release stockula 0.2.0 ([c7e6c33](https://github.com/mkm29/stockula/commit/c7e6c33a14c78307c234c23cc4ca52dc010811f2))
- **main:** release stockula 0.3.0 ([45922f8](https://github.com/mkm29/stockula/commit/45922f84f7ca86183f8c6b1cb60d21913f3135f3))
- remove branch triggers for main in Docker and Release workflows ([c258b4e](https://github.com/mkm29/stockula/commit/c258b4ef1a8e41eaa8fb95d4fc7f40f8613be188))
- remove branch triggers for main in Docker and Release workflows ([7074d82](https://github.com/mkm29/stockula/commit/7074d82544f45156970e02dcc6dc7d5cf460283e))
- reset version to v0.1.0 for initial release ([036fa12](https://github.com/mkm29/stockula/commit/036fa12646f68a386c82b38dff7ec28766d344cd))
- Update .gitignore to include AutoTS cache and sqlite database files ([6117bd3](https://github.com/mkm29/stockula/commit/6117bd30f6d55fcf62b798eb2cf9ded29300cfb3))
- update changelog with test suite fixes and improvements; modif… ([cf734ea](https://github.com/mkm29/stockula/commit/cf734ea913c9819b78da6b5b2dcb6e1b4bedf44a))
- update changelog with test suite fixes and improvements; modify Dockerfile version; enhance testing documentation; add backward compatibility methods in DatabaseManager ([8555fc6](https://github.com/mkm29/stockula/commit/8555fc60e54affe6956cf18654b2b0f41352f315))
- updated .dockerignore in order to build uv project ([a4eb5da](https://github.com/mkm29/stockula/commit/a4eb5daf64c9bdcae3529f9864ac7adf35e96214))
- updated docker-build job to add build time labels ([71ecc8b](https://github.com/mkm29/stockula/commit/71ecc8b488e6b79b7550738894cf9826b0e4d90b))

## [0.2.0](https://github.com/mkm29/stockula/compare/v0.1.0...v0.2.0) (2025-07-31)

### Features

- Add initial versioning configuration and changelog documentation ([b134945](https://github.com/mkm29/stockula/commit/b134945a74a48d472e7859ef9a5879ff332a89c8))
- add pytest-xdist for parallel test execution and improve test d… ([e07846a](https://github.com/mkm29/stockula/commit/e07846a4a5ece26ad576fa5d81e2119bf4d059c7))
- add pytest-xdist for parallel test execution and improve test database handling ([006a288](https://github.com/mkm29/stockula/commit/006a288ebe50754418395f720a6cec4c7a1e1cbb))
- added pre-commit hooks to lint code, check data types ([4e577fb](https://github.com/mkm29/stockula/commit/4e577fb442e7ed24c618f8f8117b905d0d84e887))
- **config:** Add release configuration and manifest for version 0.2.… ([08486a4](https://github.com/mkm29/stockula/commit/08486a40bc9cbf5e593a56aa6f8bee8e89afcd7f))
- **config:** Add release configuration and manifest for version 0.2.0; update .gitignore to exclude JSON files ([7a21aff](https://github.com/mkm29/stockula/commit/7a21affad0efb0e4347452821f09c494265c709a))
- **docs:** Add development documentation for AutoTS threading considerations ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- **docs:** Update changelog and user guide for train/test evaluation enhancements ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- **docs:** Update changelog and user guide for train/test evaluation… ([82c2fbb](https://github.com/mkm29/stockula/commit/82c2fbbce7eb0ae10863ed3348e5125cd2c3f60e))
- **docs:** Update changelog for version 0.2.0; enhance forecast evaluation and backtest functionality ([e061515](https://github.com/mkm29/stockula/commit/e061515b033024d0ee2c3ac9e609a035b00b09eb))
- **domain:** Exclude tickers with 0% allocation from portfolio creation and tests ([ac987dc](https://github.com/mkm29/stockula/commit/ac987dc3dccd2ae83c5b9f6552dbca0e92a4af8e))
- Enhance main.py with portfolio summary and initial value calculations ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- **forecasting:** Implement train/test split evaluation for stock price forecasting ([ce74bd6](https://github.com/mkm29/stockula/commit/ce74bd64b610ac2e6b8dc1407f2dcc8864a77e24))
- Implement Ticker domain model with singleton registry ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Initialize SQLite database for stock data storage ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- **logging:** Add method to set module-specific logging levels ([8b95356](https://github.com/mkm29/stockula/commit/8b953561b30c73f58bd0d270e7b887f688a762e0))
- **logging:** Add method to set module-specific logging levels ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **logging:** Update LoggingManager to handle additional third-party libraries ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **main:** Add detailed portfolio holdings display in console output ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))
- **main:** Enhance forecasting output with return percentage and improve logging ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **main:** Enhance portfolio value display with forecasted values and accuracy metrics ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- update CI/CD workflows and documentation ([0e48c3d](https://github.com/mkm29/stockula/commit/0e48c3d4c4c637800e318f95276970ff611b9e8f))

### Bug Fixes

- **backtesting:** Suppress progress output in backtest runner ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- Correct import paths in main.py ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Display correct strategy-specific backtest results in portfolio summary ([6bd5a97](https://github.com/mkm29/stockula/commit/6bd5a9786e6662448d73229c387fe2c38ed1b788))
- **docs:** Update documentation links to use relative paths ([a5079ba](https://github.com/mkm29/stockula/commit/a5079bad4b6aaec35a8b19b3d8918ae0dd5e3686))
- update .gitignore and CHANGELOG for CLAUDE file renaming ([324c7e6](https://github.com/mkm29/stockula/commit/324c7e615841c48c713a03085515b7da490d79d6))
- update .gitignore and CHANGELOG for CLAUDE file renaming ([4fd14b1](https://github.com/mkm29/stockula/commit/4fd14b1cf4dde2500690d22ddb91fd8af0edc907))

### Documentation

- Add broker configuration documentation to README ([e474458](https://github.com/mkm29/stockula/commit/e47445876c2437c3fe5cea99054d587e308ea36b))
- Update README to include system requirements, dependencies, architecture overview, and data flow ([505b797](https://github.com/mkm29/stockula/commit/505b7973878f7e6fd541e12c79f02b9b61074aea))
- Update stockula.yml and stockula.yml.example with new structure ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Update storage requirement in README for data caching ([da020fa](https://github.com/mkm29/stockula/commit/da020fab7edd3e1de4a962f2620f85b4cf26f2a5))
- Update user guide and README to reflect sorting by return in analysis modes ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))

### Code Refactoring

- Create structured data models for backtest results ([d8b5fe2](https://github.com/mkm29/stockula/commit/d8b5fe2ff3a1ce63265fdf8afb5132e83a16adcf))
- Organize configuration files into examples directory ([ecc4c0b](https://github.com/mkm29/stockula/commit/ecc4c0bd62b57f47cbbe4fc5887f341cd12006ae))
- Update forecaster.py to streamline imports ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))

### Tests

- **autots:** Create direct test for AutoTS to isolate hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add minimal test configuration for two tickers ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add quick test configuration with a single ticker ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add simple test configuration for forecast testing ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add test configuration for parallel progress tracking ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **parallel:** Implement tests for parallel forecasting to debug hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **progress:** Create test script to demonstrate parallel forecasting progress tracking ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **sequential:** Implement sequential forecasting test to verify functionality ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **single:** Create test for single ticker forecasting to debug hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **unit:** Refactor main tests to use parallel forecasting and improve logging checks ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **unit:** Update unit tests for forecaster to check output suppression ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))

### Miscellaneous Chores

- Add CLAUDE.market_data and .claude to .gitignore ([0cf0065](https://github.com/mkm29/stockula/commit/0cf006507c51b49bbd8b66b165c33e28434ccb81))
- Add example configuration files for different allocation strategies ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Add SQLite database initialization and testing script ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- comment out pull_request branch triggers in workflow files ([bc9ee6f](https://github.com/mkm29/stockula/commit/bc9ee6f54cc4c1ccdcd6826345b73cafdfa6a77f))
- copy in README.md in Dockerfile ([f6a8db0](https://github.com/mkm29/stockula/commit/f6a8db0f69652d56e8e562db6c44d8ae266a93cf))
- fixed pip install command in Dockerfile ([4765432](https://github.com/mkm29/stockula/commit/4765432a8cde9a789cbb16b8ea72fd2a42d408fc))
- **main:** release stockula 0.2.0 ([e779ca3](https://github.com/mkm29/stockula/commit/e779ca31ad6122c3320b419ce4766f851053e6c8))
- **main:** release stockula 0.2.0 ([c7e6c33](https://github.com/mkm29/stockula/commit/c7e6c33a14c78307c234c23cc4ca52dc010811f2))
- **main:** release stockula 0.3.0 ([45922f8](https://github.com/mkm29/stockula/commit/45922f84f7ca86183f8c6b1cb60d21913f3135f3))
- remove branch triggers for main in Docker and Release workflows ([c258b4e](https://github.com/mkm29/stockula/commit/c258b4ef1a8e41eaa8fb95d4fc7f40f8613be188))
- remove branch triggers for main in Docker and Release workflows ([7074d82](https://github.com/mkm29/stockula/commit/7074d82544f45156970e02dcc6dc7d5cf460283e))
- reset version to v0.1.0 for initial release ([036fa12](https://github.com/mkm29/stockula/commit/036fa12646f68a386c82b38dff7ec28766d344cd))
- Update .gitignore to include AutoTS cache and sqlite database files ([6117bd3](https://github.com/mkm29/stockula/commit/6117bd30f6d55fcf62b798eb2cf9ded29300cfb3))
- update changelog with test suite fixes and improvements; modif… ([cf734ea](https://github.com/mkm29/stockula/commit/cf734ea913c9819b78da6b5b2dcb6e1b4bedf44a))
- update changelog with test suite fixes and improvements; modify Dockerfile version; enhance testing documentation; add backward compatibility methods in DatabaseManager ([8555fc6](https://github.com/mkm29/stockula/commit/8555fc60e54affe6956cf18654b2b0f41352f315))
- updated .dockerignore in order to build uv project ([a4eb5da](https://github.com/mkm29/stockula/commit/a4eb5daf64c9bdcae3529f9864ac7adf35e96214))
- updated docker-build job to add build time labels ([71ecc8b](https://github.com/mkm29/stockula/commit/71ecc8b488e6b79b7550738894cf9826b0e4d90b))

## Changelog

All notable changes to this project will be documented in this file.

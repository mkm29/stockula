# Changelog

## [0.15.3](https://github.com/mkm29/stockula/compare/v0.15.2...v0.15.3) (2025-08-11)


### Bug Fixes

* remove unnecessary SHA-based tagging for Docker image builds ([330ee1d](https://github.com/mkm29/stockula/commit/330ee1d2bab31b35744386dd903014b61e45a568))
* remove unnecessary SHA-based tagging for Docker image builds ([f4a3104](https://github.com/mkm29/stockula/commit/f4a3104085a9e425b3aa0911c9a71903e751fbcb))

## [0.15.2](https://github.com/mkm29/stockula/compare/v0.15.1...v0.15.2) (2025-08-11)


### Bug Fixes

* enhance Docker build workflow with disk space checks and cleanup for GPU builds ([05ad83f](https://github.com/mkm29/stockula/commit/05ad83f6c3de393ae4655c6167e47210ac44aca7))
* enhance Docker build workflow with disk space checks and cleanup… ([ec6cde9](https://github.com/mkm29/stockula/commit/ec6cde9b93032d3b1e6c41058f3d9ae767084c6e))

## [0.15.1](https://github.com/mkm29/stockula/compare/v0.15.0...v0.15.1) (2025-08-10)


### Bug Fixes

* update Dockerfile to use runtime image for reduced size and inst… ([9418932](https://github.com/mkm29/stockula/commit/9418932510b6b8713a9202d5bd9ca96bc236cc82))
* update Dockerfile to use runtime image for reduced size and install necessary build dependencies ([eac93cf](https://github.com/mkm29/stockula/commit/eac93cf8b62eab5e636eaa6cb1ea4f02edc0c57b))

## [0.15.0](https://github.com/mkm29/stockula/compare/v0.14.0...v0.15.0) (2025-08-10)


### Features

* add backtest optimized allocator ([7fee370](https://github.com/mkm29/stockula/commit/7fee37083c0fa22afa2ebfe845c9d98b88aecc77))
* Add backtest-optimized configuration and update main logic for saving configurations ([5d4ab46](https://github.com/mkm29/stockula/commit/5d4ab465ea800ebdc4954b4ce94946b0ebf2653b))
* add forecast_multiple_symbols_with_progress method in ForecastingManager for enhanced forecasting ([a1e89dc](https://github.com/mkm29/stockula/commit/a1e89dc15023b8840272006764fc78a0df3e7fd6))
* add hotfix workflow for PR validation and backporting ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* Add initial versioning configuration and changelog documentation ([b134945](https://github.com/mkm29/stockula/commit/b134945a74a48d472e7859ef9a5879ff332a89c8))
* add pytest-xdist for parallel test execution and improve test d… ([e07846a](https://github.com/mkm29/stockula/commit/e07846a4a5ece26ad576fa5d81e2119bf4d059c7))
* add pytest-xdist for parallel test execution and improve test database handling ([006a288](https://github.com/mkm29/stockula/commit/006a288ebe50754418395f720a6cec4c7a1e1cbb))
* added pre-commit hooks to lint code, check data types ([4e577fb](https://github.com/mkm29/stockula/commit/4e577fb442e7ed24c618f8f8117b905d0d84e887))
* **config:** Add release configuration and manifest for version 0.2.… ([08486a4](https://github.com/mkm29/stockula/commit/08486a40bc9cbf5e593a56aa6f8bee8e89afcd7f))
* **config:** Add release configuration and manifest for version 0.2.0; update .gitignore to exclude JSON files ([7a21aff](https://github.com/mkm29/stockula/commit/7a21affad0efb0e4347452821f09c494265c709a))
* create allocator class ([53cfde9](https://github.com/mkm29/stockula/commit/53cfde9d4fd524a7842fb7adae4b1120bfc49f60))
* create promote to production workflow ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* **database:** Implement strategy and strategy preset models with mi… ([88f67f6](https://github.com/mkm29/stockula/commit/88f67f6a25bb772adad8d683146ddc33d640a25c))
* **database:** Implement strategy and strategy preset models with migration handling ([07ac5a9](https://github.com/mkm29/stockula/commit/07ac5a9ecf422812b1c2c50d41114512125b3267))
* **docker:** Add GPU-enabled Dockerfile and Docker Compose configuration for Stockula ([772fccd](https://github.com/mkm29/stockula/commit/772fccdff496131a50de9a6f0657136e06124f34))
* **docker:** Enhance Docker build process for feature branches and update documentation ([689d081](https://github.com/mkm29/stockula/commit/689d0814b681f933d58bce5284993111002f64a1))
* **docker:** Remove data directory copy from GPU Dockerfile ([f85eb54](https://github.com/mkm29/stockula/commit/f85eb54009a1a62c3827efd3baa65b675d531269))
* **docker:** Update GPU-enabled Dockerfile for improved dependency management and clarity ([d455774](https://github.com/mkm29/stockula/commit/d455774e9003983d6134aab4540820c6ea36d654))
* **docs:** Add development documentation for AutoTS threading considerations ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
* **docs:** Update changelog and user guide for train/test evaluation enhancements ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
* **docs:** Update changelog and user guide for train/test evaluation… ([82c2fbb](https://github.com/mkm29/stockula/commit/82c2fbbce7eb0ae10863ed3348e5125cd2c3f60e))
* **docs:** Update changelog for version 0.2.0; enhance forecast evaluation and backtest functionality ([e061515](https://github.com/mkm29/stockula/commit/e061515b033024d0ee2c3ac9e609a035b00b09eb))
* **domain:** Exclude tickers with 0% allocation from portfolio creation and tests ([ac987dc](https://github.com/mkm29/stockula/commit/ac987dc3dccd2ae83c5b9f6552dbca0e92a4af8e))
* Enhance Docker build workflow to conditionally skip GPU builds for feature branches ([9322606](https://github.com/mkm29/stockula/commit/9322606814889ecb8c8ece9a923f062649e84230))
* enhance Docker workflows and documentation for production and C… ([43c4523](https://github.com/mkm29/stockula/commit/43c4523a2870f62aa91300d1451a6b8887eccc02))
* enhance Docker workflows and documentation for production and CLI images ([d96038d](https://github.com/mkm29/stockula/commit/d96038d4909fb9ce7c816508d013ec5bdac84eb2))
* Enhance documentation and testing coverage ([35aa9e4](https://github.com/mkm29/stockula/commit/35aa9e4ad7ed0f70ed2c00b033bf3a0b5ff87fc4))
* Enhance main.py with portfolio summary and initial value calculations ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* enhance release workflow for develop branch ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* **forecasting:** Implement train/test split evaluation for stock price forecasting ([ce74bd6](https://github.com/mkm29/stockula/commit/ce74bd64b610ac2e6b8dc1407f2dcc8864a77e24))
* Implement backtest-optimized asset allocation strategy ([ba2fe24](https://github.com/mkm29/stockula/commit/ba2fe2476aa015710621b0bf4949d6bc4df3500c))
* Implement Backtest-Optimized Asset Allocation Strategy ([b08e60f](https://github.com/mkm29/stockula/commit/b08e60f3b89c71ec15be5744b79d4f69a90c5650))
* implement run_main_processing method in StockulaManager for streamlined processing ([a1e89dc](https://github.com/mkm29/stockula/commit/a1e89dc15023b8840272006764fc78a0df3e7fd6))
* Implement Ticker domain model with singleton registry ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* Initialize SQLite database for stock data storage ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* integrate chronos ([36a42e6](https://github.com/mkm29/stockula/commit/36a42e6b8ca0191121393b9e347c2ac2952132a5))
* Integrate data fetcher into Portfolio and DomainFactory for improved data handling ([57cc2b1](https://github.com/mkm29/stockula/commit/57cc2b1ad9dbd4010f633638ea00fcec153e4f8e))
* Introduce BacktestingManager for unified backtesting interface ([dc05572](https://github.com/mkm29/stockula/commit/dc055720c5865c690bc9230073ddcf9bbe0d0849))
* Introduce ForecastingManager for enhanced forecasting capabilities ([4be81d0](https://github.com/mkm29/stockula/commit/4be81d01d2586284676fb2dc2898700ec9559be2))
* introduce StrategyRepository and UniversalRepositoryRegistry for enhanced strategy management ([e5a34bf](https://github.com/mkm29/stockula/commit/e5a34bf230f2b6e9159d79f2a9fdbc797c2800a8))
* Introduce TechnicalAnalysisManager for enhanced technical analysis capabilities ([9a23b4e](https://github.com/mkm29/stockula/commit/9a23b4e6ab68adf2ea75aac2c408c7511f575f32))
* **logging:** Add method to set module-specific logging levels ([8b95356](https://github.com/mkm29/stockula/commit/8b953561b30c73f58bd0d270e7b887f688a762e0))
* **logging:** Add method to set module-specific logging levels ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **logging:** Update LoggingManager to handle additional third-party libraries ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **main:** Add detailed portfolio holdings display in console output ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))
* **main:** Enhance forecasting output with return percentage and improve logging ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **main:** Enhance portfolio value display with forecasted values and accuracy metrics ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
* module managers ([5812251](https://github.com/mkm29/stockula/commit/581225168b516fbbb4c024622bad5d8694123f4b))
* **portfolio:** Enhance portfolio creation with validation error handling and suggestions ([0f63341](https://github.com/mkm29/stockula/commit/0f633418aa688e58c9c46ebf0bc6bad3d92dc172))
* Refactor portfolio creation to use dependency injection and remove obsolete model ([7b588e5](https://github.com/mkm29/stockula/commit/7b588e5c61b1e539d69d6449e2a1399fde48f3a1))
* **tests:** Update integration and unit test workflows with coverage reporting and cleanup steps ([4043ab0](https://github.com/mkm29/stockula/commit/4043ab0684de94bd66c28af53a5ccd49e9787538))
* update branch protection permissions and fix portfolio creation in integration tests ([5c44ad0](https://github.com/mkm29/stockula/commit/5c44ad047b39da05274d28a530fc9ff5dbebf288))
* Update branch protection rules and enhance dependency injection in the container ([b335eb5](https://github.com/mkm29/stockula/commit/b335eb5f731c3b18dc922932a789b2635ef91705))
* Update branch protection rules to include Release Please branches for merging to develop ([5b2e3b0](https://github.com/mkm29/stockula/commit/5b2e3b0a070a2bfe344009e406daf03f741144a8))
* update CI/CD workflows and documentation ([0e48c3d](https://github.com/mkm29/stockula/commit/0e48c3d4c4c637800e318f95276970ff611b9e8f))


### Bug Fixes

* add chronos availability check and update ForecastingManager import logic ([58d1d9e](https://github.com/mkm29/stockula/commit/58d1d9efe0c288dc0a9609a134dd894257c1aa75))
* add UV_LINK_MODE environment variable for improved bytecode handling ([45c3d97](https://github.com/mkm29/stockula/commit/45c3d970debcc12475051c51dcd96d8f32c50db1))
* added disk space cleanup acton ([1496631](https://github.com/mkm29/stockula/commit/1496631bb3e8d861e7fb84b5b1b39ce72c6c0b2f))
* added disk space cleanup acton ([0b5278b](https://github.com/mkm29/stockula/commit/0b5278bf820a4062c5f6cdaa139b13ce411f3a9c))
* adjust versioning ([1db4132](https://github.com/mkm29/stockula/commit/1db41329608859e281c24c03923472598ec64971))
* **backtesting:** Suppress progress output in backtest runner ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
* changed the tagging of docker images in docker-build.yml ([71f17e6](https://github.com/mkm29/stockula/commit/71f17e600160646609aacfe0d715742eaa4ddcaf))
* correct GPU acceleration documentation for clarity ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* Correct import paths in main.py ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* Display correct strategy-specific backtest results in portfolio summary ([6bd5a97](https://github.com/mkm29/stockula/commit/6bd5a9786e6662448d73229c387fe2c38ed1b788))
* **docs:** Update documentation links to use relative paths ([a5079ba](https://github.com/mkm29/stockula/commit/a5079bad4b6aaec35a8b19b3d8918ae0dd5e3686))
* Enhance documentation and testing coverage ([3ffcc96](https://github.com/mkm29/stockula/commit/3ffcc967594a3f47aae83100bc9eb9dc89ae5a3e))
* Improve formatting of coverage status tables in documentation ([69f7a75](https://github.com/mkm29/stockula/commit/69f7a75d375497cbe4c9a3f30e41f8883836c654))
* improve test coverage ([f19dffe](https://github.com/mkm29/stockula/commit/f19dffe92aa584ff1260a7e91f46fde692dabf09))
* improve test coverage ([cfeafd1](https://github.com/mkm29/stockula/commit/cfeafd1caf2e2b78d1f90e7ed357149855b53410))
* improve YAML formatting documentation ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* introduce StrategyRepository and UniversalRepositoryRegistry fo… ([26f87eb](https://github.com/mkm29/stockula/commit/26f87eb281bfbf962f682b1de187d2bbce6f3fa8))
* optimized Dockerfile and build workflow for running on GitHub Ac… ([f9f52ac](https://github.com/mkm29/stockula/commit/f9f52acb195a65dbbb8b3710491ae78cc1920ff9))
* optimized Dockerfile and build workflow for running on GitHub Actions ([13106b6](https://github.com/mkm29/stockula/commit/13106b6bd2c7428ef16b655ed515b3b1cde8cdc6))
* reduced logic in main.py ([0ac38b5](https://github.com/mkm29/stockula/commit/0ac38b5effe74dab82b8f784ea081e6a323f8c64))
* Refactor tests and main CLI functionality ([109397e](https://github.com/mkm29/stockula/commit/109397e1711caf71d302d3e7f319bcfbb359d6b0))
* remove obsolete test for repository patterns from workflow ([af0668f](https://github.com/mkm29/stockula/commit/af0668f0913c34053bc07209fb9918bef2948f37))
* removed autots tests from .github/workflows/test.yaml ([e61bfc3](https://github.com/mkm29/stockula/commit/e61bfc37a43c5a7a584dee901836a37f4893581a))
* removed unused packages from pyproject.toml ([cf2abac](https://github.com/mkm29/stockula/commit/cf2abacd83f6eec1a20f47c3cc3dad4a021b48a9))
* removed unused packages from pyproject.toml ([8bf22e1](https://github.com/mkm29/stockula/commit/8bf22e189b0b0167399c5f415464fa5918668884))
* Rename loop variable for clarity and remove unused mock strategy ([a0209d9](https://github.com/mkm29/stockula/commit/a0209d9a326ca9080da4500a67d2b3a0f65d6cba))
* require successful run of test GitHub Action before docker-build is run ([d087013](https://github.com/mkm29/stockula/commit/d08701394061c9ff5c401bde3a90b73a3d3bca70))
* require successful run of test GitHub Action before docker-build… ([38e94a8](https://github.com/mkm29/stockula/commit/38e94a81a5b3afa5da6e04f891e036a6cac23585))
* resolved timezone discrepancies in backtesting ([2619b88](https://github.com/mkm29/stockula/commit/2619b8862423eabeaea1f560801062b06ce96ccf))
* resolved timezone discrepancies in backtesting ([9e95e95](https://github.com/mkm29/stockula/commit/9e95e95b2678f80da7b27aacd52f95dc4c9032ed))
* reverted to previous docker build (only build cli) ([237ee63](https://github.com/mkm29/stockula/commit/237ee63ed6ba9b3cb5fc29a6b9352a99319517e6))
* update .gitignore and CHANGELOG for CLAUDE file renaming ([324c7e6](https://github.com/mkm29/stockula/commit/324c7e615841c48c713a03085515b7da490d79d6))
* update .gitignore and CHANGELOG for CLAUDE file renaming ([4fd14b1](https://github.com/mkm29/stockula/commit/4fd14b1cf4dde2500690d22ddb91fd8af0edc907))
* Update Docker build tags to differentiate between release and no… ([7d293b2](https://github.com/mkm29/stockula/commit/7d293b22b2792c0a4eeeef262b34f50cb1792f4b))
* Update Docker build tags to differentiate between release and non-release events ([f23163d](https://github.com/mkm29/stockula/commit/f23163dc1464785203cbc0bdee3b91cdd0168406))
* update docker validation documentation for clarity ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* update integration test workflow to include additional paths ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* update README links to ensure proper navigation ([bd5e1df](https://github.com/mkm29/stockula/commit/bd5e1dff819f38d109ba041a84cd06b055bbf001))
* update README links to ensure proper navigation ([6563c19](https://github.com/mkm29/stockula/commit/6563c1949e250b21b964b264d47252457dc83d50))
* Use container's domain factory with proper dependency injection in end-to-end test ([d441136](https://github.com/mkm29/stockula/commit/d4411369b929462a39034e4fdf6fa6742d18a151))


### Documentation

* Add broker configuration documentation to README ([e474458](https://github.com/mkm29/stockula/commit/e47445876c2437c3fe5cea99054d587e308ea36b))
* create contributing guide for development workflow ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* enhance Docker deployment documentation ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* Update README to include system requirements, dependencies, architecture overview, and data flow ([505b797](https://github.com/mkm29/stockula/commit/505b7973878f7e6fd541e12c79f02b9b61074aea))
* update README with Git Flow and release process details ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* Update stockula.yml and stockula.yml.example with new structure ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* Update storage requirement in README for data caching ([da020fa](https://github.com/mkm29/stockula/commit/da020fab7edd3e1de4a962f2620f85b4cf26f2a5))
* Update user guide and README to reflect sorting by return in analysis modes ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))


### Code Refactoring

* clean up Hadolint config and enhance test coverage for configuration loading ([c25c2c3](https://github.com/mkm29/stockula/commit/c25c2c3c0c2599e1729b105889813922f2065999))
* Create structured data models for backtest results ([d8b5fe2](https://github.com/mkm29/stockula/commit/d8b5fe2ff3a1ce63265fdf8afb5132e83a16adcf))
* Organize configuration files into examples directory ([ecc4c0b](https://github.com/mkm29/stockula/commit/ecc4c0bd62b57f47cbbe4fc5887f341cd12006ae))
* streamline test workflow by removing integration tests ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* Update forecaster.py to streamline imports ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))


### Tests

* **autots:** Create direct test for AutoTS to isolate hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **config:** Add minimal test configuration for two tickers ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **config:** Add quick test configuration with a single ticker ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **config:** Add simple test configuration for forecast testing ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **config:** Add test configuration for parallel progress tracking ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **parallel:** Implement tests for parallel forecasting to debug hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **progress:** Create test script to demonstrate parallel forecasting progress tracking ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **sequential:** Implement sequential forecasting test to verify functionality ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **single:** Create test for single ticker forecasting to debug hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **unit:** Refactor main tests to use parallel forecasting and improve logging checks ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **unit:** Update unit tests for forecaster to check output suppression ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* update unit tests to reflect changes in main processing and forecasting methods ([a1e89dc](https://github.com/mkm29/stockula/commit/a1e89dc15023b8840272006764fc78a0df3e7fd6))


### Miscellaneous Chores

* Add CLAUDE.market_data and .claude to .gitignore ([0cf0065](https://github.com/mkm29/stockula/commit/0cf006507c51b49bbd8b66b165c33e28434ccb81))
* Add example configuration files for different allocation strategies ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* Add SQLite database initialization and testing script ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* comment out pull_request branch triggers in workflow files ([bc9ee6f](https://github.com/mkm29/stockula/commit/bc9ee6f54cc4c1ccdcd6826345b73cafdfa6a77f))
* copy in README.md in Dockerfile ([f6a8db0](https://github.com/mkm29/stockula/commit/f6a8db0f69652d56e8e562db6c44d8ae266a93cf))
* **develop:** release 0.12.0 ([21845fb](https://github.com/mkm29/stockula/commit/21845fb5815f13776b9ee63c0baad0e486b8ec03))
* **develop:** release 0.12.0 ([56929ac](https://github.com/mkm29/stockula/commit/56929acd719103cdf57c1007a0fee9b223e9d7af))
* **develop:** release 0.13.0 ([bbedafa](https://github.com/mkm29/stockula/commit/bbedafa693b3a8e81c1236e6581884a8fba7ec75))
* **develop:** release 0.13.0 ([edde3fe](https://github.com/mkm29/stockula/commit/edde3feb7e9b2f4754f3174c0a5b1e094b5517cc))
* **develop:** release 0.13.1 ([aadd2c9](https://github.com/mkm29/stockula/commit/aadd2c9d59ca0265dbdc7b60bbac297fb33d8bbe))
* **develop:** release 0.13.1 ([1ca7e57](https://github.com/mkm29/stockula/commit/1ca7e571f5ea0461b9f0baf2cffd78e714a31022))
* downgrade release version to 0.2.0 in manifest ([0a69553](https://github.com/mkm29/stockula/commit/0a6955318686ea8675f3e0c95e308c3d8315ea8c))
* fixed pip install command in  Dockerfile ([4765432](https://github.com/mkm29/stockula/commit/4765432a8cde9a789cbb16b8ea72fd2a42d408fc))
* **main:** release 0.10.0 ([7d1f841](https://github.com/mkm29/stockula/commit/7d1f84100866bc4a3c125cfad76e4dc044c0233a))
* **main:** release 0.10.0 ([c7c589e](https://github.com/mkm29/stockula/commit/c7c589ef58b179cdd40a392cdc740d6cc05e03b4))
* **main:** release 0.11.0 ([22821c4](https://github.com/mkm29/stockula/commit/22821c49ff26383e457ddecc828e33f95fe2ffc6))
* **main:** release 0.11.0 ([a379879](https://github.com/mkm29/stockula/commit/a37987951ba5594314c759a582ff96bf99e00ddd))
* **main:** release 0.2.0 ([dd76ab5](https://github.com/mkm29/stockula/commit/dd76ab5a102148ac6c6c2d80481b03ec348a1786))
* **main:** release 0.2.0 ([5fe8f0e](https://github.com/mkm29/stockula/commit/5fe8f0e9ddab552546f5843e8bd6695c782799d0))
* **main:** release 0.3.0 ([9d63485](https://github.com/mkm29/stockula/commit/9d63485d45e4d4a4b8c764c3df60bc60a4dd2d1d))
* **main:** release 0.3.0 ([73c293a](https://github.com/mkm29/stockula/commit/73c293a316c8a6c7cf295bcc7d03507fe1fe9afc))
* **main:** release 0.3.1 ([51f31ef](https://github.com/mkm29/stockula/commit/51f31ef27ce1dad7a9c3151f1fcea83a77fd1f1f))
* **main:** release 0.3.1 ([3ae040e](https://github.com/mkm29/stockula/commit/3ae040e7ed65d46f3243b12ba726239511251ec2))
* **main:** release 0.3.2 ([fbc598b](https://github.com/mkm29/stockula/commit/fbc598b3630a50665e4fe971131b212bea2fcc44))
* **main:** release 0.3.2 ([2438a5d](https://github.com/mkm29/stockula/commit/2438a5db6f8e7c8565945c31fd141f1c76303f54))
* **main:** release 0.4.0 ([8f5270d](https://github.com/mkm29/stockula/commit/8f5270d393ec9505d5e2723c8f2466631d27a3d3))
* **main:** release 0.4.0 ([e869979](https://github.com/mkm29/stockula/commit/e869979eb3e9ab16f3f3b1eba7fb78f9f929d08d))
* **main:** release 0.4.1 ([4322a70](https://github.com/mkm29/stockula/commit/4322a7003b1d562656c63b2a6789411fee7ad6b0))
* **main:** release 0.4.1 ([963484e](https://github.com/mkm29/stockula/commit/963484e1c48cc06170a1b593fc6bf7ccd7cfa6fb))
* **main:** release 0.4.2 ([f44d58e](https://github.com/mkm29/stockula/commit/f44d58ea8fcf0c51e228e7ddc704149305b0f776))
* **main:** release 0.4.2 ([7d80ac3](https://github.com/mkm29/stockula/commit/7d80ac387c37ae4c196c73f41a4a7261b30146e0))
* **main:** release 0.4.3 ([04b9109](https://github.com/mkm29/stockula/commit/04b910990d8c6a9d02bec3625c846acf40838ace))
* **main:** release 0.4.3 ([cd4c642](https://github.com/mkm29/stockula/commit/cd4c64219a41df85137494fcecb6df01842dc99a))
* **main:** release 0.5.0 ([a7f67c9](https://github.com/mkm29/stockula/commit/a7f67c90b952e18519f1cde3a7e8e429fa25791b))
* **main:** release 0.5.0 ([5d419c1](https://github.com/mkm29/stockula/commit/5d419c115a64c2d690bdffc63defdc95db272138))
* **main:** release 0.5.1 ([0e0071d](https://github.com/mkm29/stockula/commit/0e0071d2bcbdd4b6b59a9615b06e64cead1a075a))
* **main:** release 0.5.1 ([beac662](https://github.com/mkm29/stockula/commit/beac6622a70d8af2d8c7910e15db4072e4feabe3))
* **main:** release 0.5.2 ([845cdda](https://github.com/mkm29/stockula/commit/845cddaee3fdf530cc255b60dd17cee15131a73f))
* **main:** release 0.5.2 ([478cc6e](https://github.com/mkm29/stockula/commit/478cc6eed4d1fa70697787fda8110b7be0157759))
* **main:** release 0.6.0 ([474a37e](https://github.com/mkm29/stockula/commit/474a37e5fc4d85254f7bca4c81fa90fcfaad4f63))
* **main:** release 0.6.0 ([d1c719f](https://github.com/mkm29/stockula/commit/d1c719f28e2207cc30a875a9ddbfd29df3affde0))
* **main:** release 0.6.1 ([5c4a0d9](https://github.com/mkm29/stockula/commit/5c4a0d935cbe6a2a7ccf623ab31c3d02f8b5edf0))
* **main:** release 0.6.1 ([e3aaa75](https://github.com/mkm29/stockula/commit/e3aaa7552ea112d04997f1bd2025837788837bc1))
* **main:** release 0.7.0 ([4f3b91a](https://github.com/mkm29/stockula/commit/4f3b91aa28253f38f6307087d8e7337d7a3fb171))
* **main:** release 0.7.0 ([64d9317](https://github.com/mkm29/stockula/commit/64d93173c4c094a99ca2f9eb7eb865d7b0dba026))
* **main:** release 0.8.0 ([94a4281](https://github.com/mkm29/stockula/commit/94a428148d897aa870d08717db717277612313c2))
* **main:** release 0.8.0 ([6af9e1a](https://github.com/mkm29/stockula/commit/6af9e1ab9808112aa08cd2be0d1aee8908a6b001))
* **main:** release 0.9.0 ([7e1e81c](https://github.com/mkm29/stockula/commit/7e1e81c9641b9415442df33bdf979ad410280e38))
* **main:** release 0.9.0 ([629029d](https://github.com/mkm29/stockula/commit/629029d67c9fe302775f36a7864b2e264472b4bf))
* **main:** release 0.9.1 ([478eb46](https://github.com/mkm29/stockula/commit/478eb46db1f698958e2e9636ba2551f6ec5a92bf))
* **main:** release 0.9.1 ([c2d408f](https://github.com/mkm29/stockula/commit/c2d408f284c6f5f7205c48b06b0ebdeca0ca3b49))
* **main:** release stockula 0.2.0 ([e779ca3](https://github.com/mkm29/stockula/commit/e779ca31ad6122c3320b419ce4766f851053e6c8))
* **main:** release stockula 0.2.0 ([c7e6c33](https://github.com/mkm29/stockula/commit/c7e6c33a14c78307c234c23cc4ca52dc010811f2))
* **main:** release stockula 0.3.0 ([45922f8](https://github.com/mkm29/stockula/commit/45922f84f7ca86183f8c6b1cb60d21913f3135f3))
* remove branch triggers for main in Docker and Release workflows ([c258b4e](https://github.com/mkm29/stockula/commit/c258b4ef1a8e41eaa8fb95d4fc7f40f8613be188))
* remove branch triggers for main in Docker and Release workflows ([7074d82](https://github.com/mkm29/stockula/commit/7074d82544f45156970e02dcc6dc7d5cf460283e))
* reset version to v0.1.0 for initial release ([036fa12](https://github.com/mkm29/stockula/commit/036fa12646f68a386c82b38dff7ec28766d344cd))
* Update .gitignore to include AutoTS cache and sqlite database files ([6117bd3](https://github.com/mkm29/stockula/commit/6117bd30f6d55fcf62b798eb2cf9ded29300cfb3))
* update changelog with test suite fixes and improvements; modif… ([cf734ea](https://github.com/mkm29/stockula/commit/cf734ea913c9819b78da6b5b2dcb6e1b4bedf44a))
* update changelog with test suite fixes and improvements; modify Dockerfile version; enhance testing documentation; add backward compatibility methods in DatabaseManager ([8555fc6](https://github.com/mkm29/stockula/commit/8555fc60e54affe6956cf18654b2b0f41352f315))
* update Codecov action to version 5 in test workflow ([7e985ac](https://github.com/mkm29/stockula/commit/7e985ac1d1a892b1e9c425a05a29c7f491fcc209))
* update linting commands for consistency with CI and improve documentation ([d02ffa7](https://github.com/mkm29/stockula/commit/d02ffa7207109987fa0a6d7967fae60d61759f36))
* update release-please configuration for prerelease tagging ([a6092aa](https://github.com/mkm29/stockula/commit/a6092aaa483c3bef61da100e91db0ea538aa72e0))
* Update test workflow to simplify unit test command and add JUnit XML reporting ([55aa3a1](https://github.com/mkm29/stockula/commit/55aa3a11e23d7b86d2136c9e10b109a3ed53bd7b))
* update version to 0.14.0 and adjust Python version to 3.11 in workflows and configuration files; add chronos availability check in tests ([5cbfc84](https://github.com/mkm29/stockula/commit/5cbfc842ce7dcd4c4267baca01f4e9c8d959b37b))
* update version to 0.8.0 in uv.lock and modify CHANGELOG for new features ([a1e89dc](https://github.com/mkm29/stockula/commit/a1e89dc15023b8840272006764fc78a0df3e7fd6))
* updated .dockerignore in order to build uv project ([a4eb5da](https://github.com/mkm29/stockula/commit/a4eb5daf64c9bdcae3529f9864ac7adf35e96214))
* updated docker-build job to add build time labels ([71ecc8b](https://github.com/mkm29/stockula/commit/71ecc8b488e6b79b7550738894cf9826b0e4d90b))

## [0.13.1](https://github.com/mkm29/stockula/compare/v0.13.0...v0.13.1) (2025-08-10)

### Bug Fixes

- remove obsolete test for repository patterns from workflow
  ([af0668f](https://github.com/mkm29/stockula/commit/af0668f0913c34053bc07209fb9918bef2948f37))
- removed autots tests from .github/workflows/test.yaml
  ([e61bfc3](https://github.com/mkm29/stockula/commit/e61bfc37a43c5a7a584dee901836a37f4893581a))

## [Unreleased]

### Changed

- **BREAKING**: Completely removed AutoTS dependency in favor of AutoGluon TimeSeries
- **BREAKING**: Removed `StockForecaster` class - use `ForecastingManager` instead
- **BREAKING**: Removed AutoTS-specific configuration options (model_list, ensemble, max_generations, etc.)
- Migrated forecasting to AutoGluon TimeSeries with automatic fallback to simple LinearRegression for Python 3.13+
- Simplified ForecastConfig to only include AutoGluon-specific settings
- Updated all documentation to reflect AutoGluon-only forecasting

### Added

- `SimpleForecastBackend` using LinearRegression for environments without AutoGluon
- Automatic backend selection based on Python version and package availability
- Support for Python 3.13+ through fallback forecasting implementation

### Removed

- AutoTS dependency and all related code
- `StockForecaster` class and tests
- AutoTS model validation and repository classes
- AutoTS-specific documentation and examples

### Previous Changes

- **BREAKING**: Removed `AutoTSModelValidator` class - model validation is now database-driven
- Refactored AutoTS model validation to use database instead of hardcoded lists
- AutoTSModel and AutoTSPreset are now SQLModel-based database models with self-validation
- Model validation is performed automatically when saving to database

### Added

- `AutoTSRepository` for managing AutoTS models and presets in the database
- Database seeding directly from AutoTS library when database is empty
- Comprehensive test coverage for `autots_repository.py` (95% coverage)
- Complete test coverage for `backtesting/registry.py` (100% coverage)
- Complete test coverage for `config/exceptions.py` (100% coverage)
- Self-validating domain models with database persistence

### Improved

- Overall test coverage increased from 21% to 86%
- Dependency injection for DatabaseManager in forecasting components
- More flexible model management without code changes

## [0.13.0](https://github.com/mkm29/stockula/compare/v0.12.0...v0.13.0) (2025-08-09)

### Features

- Update branch protection rules to include Release Please branches for merging to develop
  ([5b2e3b0](https://github.com/mkm29/stockula/commit/5b2e3b0a070a2bfe344009e406daf03f741144a8))

### Bug Fixes

- Update Docker build tags to differentiate between release and no…
  ([7d293b2](https://github.com/mkm29/stockula/commit/7d293b22b2792c0a4eeeef262b34f50cb1792f4b))
- Update Docker build tags to differentiate between release and non-release events
  ([f23163d](https://github.com/mkm29/stockula/commit/f23163dc1464785203cbc0bdee3b91cdd0168406))

## [0.12.0](https://github.com/mkm29/stockula/compare/v0.11.0...v0.12.0) (2025-08-09)

### Features

- **docker:** Enhance Docker build process for feature branches and update documentation
  ([689d081](https://github.com/mkm29/stockula/commit/689d0814b681f933d58bce5284993111002f64a1))
- **docker:** Remove data directory copy from GPU Dockerfile
  ([f85eb54](https://github.com/mkm29/stockula/commit/f85eb54009a1a62c3827efd3baa65b675d531269))
- Enhance Docker build workflow to conditionally skip GPU builds for feature branches
  ([9322606](https://github.com/mkm29/stockula/commit/9322606814889ecb8c8ece9a923f062649e84230))
- Integrate data fetcher into Portfolio and DomainFactory for improved data handling
  ([57cc2b1](https://github.com/mkm29/stockula/commit/57cc2b1ad9dbd4010f633638ea00fcec153e4f8e))
- Refactor portfolio creation to use dependency injection and remove obsolete model
  ([7b588e5](https://github.com/mkm29/stockula/commit/7b588e5c61b1e539d69d6449e2a1399fde48f3a1))
- update branch protection permissions and fix portfolio creation in integration tests
  ([5c44ad0](https://github.com/mkm29/stockula/commit/5c44ad047b39da05274d28a530fc9ff5dbebf288))
- Update branch protection rules and enhance dependency injection in the container
  ([b335eb5](https://github.com/mkm29/stockula/commit/b335eb5f731c3b18dc922932a789b2635ef91705))

### Bug Fixes

- Use container's domain factory with proper dependency injection in end-to-end test
  ([d441136](https://github.com/mkm29/stockula/commit/d4411369b929462a39034e4fdf6fa6742d18a151))

## [0.10.0](https://github.com/mkm29/stockula/compare/v0.9.1...v0.10.0) (2025-08-05)

### Features

- introduce StrategyRepository and UniversalRepositoryRegistry for enhanced strategy management
  ([e5a34bf](https://github.com/mkm29/stockula/commit/e5a34bf230f2b6e9159d79f2a9fdbc797c2800a8))

### Bug Fixes

- introduce StrategyRepository and UniversalRepositoryRegistry fo…
  ([26f87eb](https://github.com/mkm29/stockula/commit/26f87eb281bfbf962f682b1de187d2bbce6f3fa8))

## [0.9.1](https://github.com/mkm29/stockula/compare/v0.9.0...v0.9.1) (2025-08-04)

### Bug Fixes

- Refactor tests and main CLI functionality
  ([109397e](https://github.com/mkm29/stockula/commit/109397e1711caf71d302d3e7f319bcfbb359d6b0))

## [0.9.0](https://github.com/mkm29/stockula/compare/v0.8.0...v0.9.0) (2025-08-04)

### Features

- add forecast_multiple_symbols_with_progress method in ForecastingManager for enhanced forecasting
  ([a1e89dc](https://github.com/mkm29/stockula/commit/a1e89dc15023b8840272006764fc78a0df3e7fd6))
- implement run_main_processing method in StockulaManager for streamlined processing
  ([a1e89dc](https://github.com/mkm29/stockula/commit/a1e89dc15023b8840272006764fc78a0df3e7fd6))

### Bug Fixes

- reduced logic in main.py
  ([0ac38b5](https://github.com/mkm29/stockula/commit/0ac38b5effe74dab82b8f784ea081e6a323f8c64))

### Tests

- update unit tests to reflect changes in main processing and forecasting methods
  ([a1e89dc](https://github.com/mkm29/stockula/commit/a1e89dc15023b8840272006764fc78a0df3e7fd6))

### Miscellaneous Chores

- update version to 0.8.0 in uv.lock and modify CHANGELOG for new features
  ([a1e89dc](https://github.com/mkm29/stockula/commit/a1e89dc15023b8840272006764fc78a0df3e7fd6))

## [0.8.0](https://github.com/mkm29/stockula/compare/v0.7.0...v0.8.0) (2025-08-04)

### Features

- Introduce BacktestingManager for unified backtesting interface
  ([dc05572](https://github.com/mkm29/stockula/commit/dc055720c5865c690bc9230073ddcf9bbe0d0849))
- Introduce ForecastingManager for enhanced forecasting capabilities
  ([4be81d0](https://github.com/mkm29/stockula/commit/4be81d01d2586284676fb2dc2898700ec9559be2))
- Introduce TechnicalAnalysisManager for enhanced technical analysis capabilities
  ([9a23b4e](https://github.com/mkm29/stockula/commit/9a23b4e6ab68adf2ea75aac2c408c7511f575f32))
- module managers ([5812251](https://github.com/mkm29/stockula/commit/581225168b516fbbb4c024622bad5d8694123f4b))
- Replace argparse with Typer for modern CLI interface
- Extract CLI logic to dedicated cli.py module for better separation of concerns
- Refactor main.py logic into managers for improved code organization

### Bug Fixes

- Improve formatting of coverage status tables in documentation
  ([69f7a75](https://github.com/mkm29/stockula/commit/69f7a75d375497cbe4c9a3f30e41f8883836c654))
- Fix B008 linting errors in CLI module by using Annotated type syntax

### Code Refactoring

- Move business logic from main.py to StockulaManager
- Add progress tracking methods to ForecastingManager
- Reduce main.py from ~300 lines to 53 lines

## [0.7.0](https://github.com/mkm29/stockula/compare/v0.6.1...v0.7.0) (2025-08-03)

### Features

- add backtest optimized allocator
  ([7fee370](https://github.com/mkm29/stockula/commit/7fee37083c0fa22afa2ebfe845c9d98b88aecc77))
- Add backtest-optimized configuration and update main logic for saving configurations
  ([5d4ab46](https://github.com/mkm29/stockula/commit/5d4ab465ea800ebdc4954b4ce94946b0ebf2653b))
- Implement backtest-optimized asset allocation strategy
  ([ba2fe24](https://github.com/mkm29/stockula/commit/ba2fe2476aa015710621b0bf4949d6bc4df3500c))
- Implement Backtest-Optimized Asset Allocation Strategy
  ([b08e60f](https://github.com/mkm29/stockula/commit/b08e60f3b89c71ec15be5744b79d4f69a90c5650))

### Bug Fixes

- Rename loop variable for clarity and remove unused mock strategy
  ([a0209d9](https://github.com/mkm29/stockula/commit/a0209d9a326ca9080da4500a67d2b3a0f65d6cba))

## [0.6.1](https://github.com/mkm29/stockula/compare/v0.6.0...v0.6.1) (2025-08-02)

### Bug Fixes

- resolved timezone discrepancies in backtesting
  ([2619b88](https://github.com/mkm29/stockula/commit/2619b8862423eabeaea1f560801062b06ce96ccf))
- resolved timezone discrepancies in backtesting
  ([9e95e95](https://github.com/mkm29/stockula/commit/9e95e95b2678f80da7b27aacd52f95dc4c9032ed))

## [0.6.0](https://github.com/mkm29/stockula/compare/v0.5.2...v0.6.0) (2025-08-02)

### Features

- create allocator class ([53cfde9](https://github.com/mkm29/stockula/commit/53cfde9d4fd524a7842fb7adae4b1120bfc49f60))

### Miscellaneous Chores

- update linting commands for consistency with CI and improve documentation
  ([d02ffa7](https://github.com/mkm29/stockula/commit/d02ffa7207109987fa0a6d7967fae60d61759f36))

## [0.5.2](https://github.com/mkm29/stockula/compare/v0.5.1...v0.5.2) (2025-08-02)

### Bug Fixes

- require successful run of test GitHub Action before docker-build is run
  ([d087013](https://github.com/mkm29/stockula/commit/d08701394061c9ff5c401bde3a90b73a3d3bca70))
- require successful run of test GitHub Action before docker-build…
  ([38e94a8](https://github.com/mkm29/stockula/commit/38e94a81a5b3afa5da6e04f891e036a6cac23585))

## [0.5.1](https://github.com/mkm29/stockula/compare/v0.5.0...v0.5.1) (2025-08-02)

### Bug Fixes

- removed unused packages from pyproject.toml
  ([cf2abac](https://github.com/mkm29/stockula/commit/cf2abacd83f6eec1a20f47c3cc3dad4a021b48a9))
- removed unused packages from pyproject.toml
  ([8bf22e1](https://github.com/mkm29/stockula/commit/8bf22e189b0b0167399c5f415464fa5918668884))

## [0.5.0](https://github.com/mkm29/stockula/compare/v0.4.3...v0.5.0) (2025-08-02)

### Features

- Enhance documentation and testing coverage
  ([35aa9e4](https://github.com/mkm29/stockula/commit/35aa9e4ad7ed0f70ed2c00b033bf3a0b5ff87fc4))

### Bug Fixes

- Enhance documentation and testing coverage
  ([3ffcc96](https://github.com/mkm29/stockula/commit/3ffcc967594a3f47aae83100bc9eb9dc89ae5a3e))
- improve test coverage ([f19dffe](https://github.com/mkm29/stockula/commit/f19dffe92aa584ff1260a7e91f46fde692dabf09))

### Code Refactoring

- clean up Hadolint config and enhance test coverage for configuration loading
  ([c25c2c3](https://github.com/mkm29/stockula/commit/c25c2c3c0c2599e1729b105889813922f2065999))

### Miscellaneous Chores

- Update test workflow to simplify unit test command and add JUnit XML reporting
  ([55aa3a1](https://github.com/mkm29/stockula/commit/55aa3a11e23d7b86d2136c9e10b109a3ed53bd7b))

## [0.4.3](https://github.com/mkm29/stockula/compare/v0.4.2...v0.4.3) (2025-08-01)

### Bug Fixes

- improve test coverage ([cfeafd1](https://github.com/mkm29/stockula/commit/cfeafd1caf2e2b78d1f90e7ed357149855b53410))

## [0.4.2](https://github.com/mkm29/stockula/compare/v0.4.1...v0.4.2) (2025-08-01)

### Bug Fixes

- changed the tagging of docker images in docker-build.yml
  ([71f17e6](https://github.com/mkm29/stockula/commit/71f17e600160646609aacfe0d715742eaa4ddcaf))
- reverted to previous docker build (only build cli)
  ([237ee63](https://github.com/mkm29/stockula/commit/237ee63ed6ba9b3cb5fc29a6b9352a99319517e6))

## [0.4.1](https://github.com/mkm29/stockula/compare/v0.4.0...v0.4.1) (2025-08-01)

### Bug Fixes

- added disk space cleanup acton
  ([1496631](https://github.com/mkm29/stockula/commit/1496631bb3e8d861e7fb84b5b1b39ce72c6c0b2f))
- added disk space cleanup acton
  ([0b5278b](https://github.com/mkm29/stockula/commit/0b5278bf820a4062c5f6cdaa139b13ce411f3a9c))

## [0.4.0](https://github.com/mkm29/stockula/compare/v0.3.2...v0.4.0) (2025-08-01)

### Features

- enhance Docker workflows and documentation for production and C…
  ([43c4523](https://github.com/mkm29/stockula/commit/43c4523a2870f62aa91300d1451a6b8887eccc02))
- enhance Docker workflows and documentation for production and CLI images
  ([d96038d](https://github.com/mkm29/stockula/commit/d96038d4909fb9ce7c816508d013ec5bdac84eb2))

## [0.3.2](https://github.com/mkm29/stockula/compare/v0.3.1...v0.3.2) (2025-08-01)

### Bug Fixes

- optimized Dockerfile and build workflow for running on GitHub Ac…
  ([f9f52ac](https://github.com/mkm29/stockula/commit/f9f52acb195a65dbbb8b3710491ae78cc1920ff9))
- optimized Dockerfile and build workflow for running on GitHub Actions
  ([13106b6](https://github.com/mkm29/stockula/commit/13106b6bd2c7428ef16b655ed515b3b1cde8cdc6))

## [0.3.1](https://github.com/mkm29/stockula/compare/v0.3.0...v0.3.1) (2025-08-01)

### Bug Fixes

- adjust versioning ([1db4132](https://github.com/mkm29/stockula/commit/1db41329608859e281c24c03923472598ec64971))

### Miscellaneous Chores

- downgrade release version to 0.2.0 in manifest
  ([0a69553](https://github.com/mkm29/stockula/commit/0a6955318686ea8675f3e0c95e308c3d8315ea8c))
- update Codecov action to version 5 in test workflow
  ([7e985ac](https://github.com/mkm29/stockula/commit/7e985ac1d1a892b1e9c425a05a29c7f491fcc209))

## [0.3.0](https://github.com/mkm29/stockula/compare/v0.2.1...v0.3.0) (2025-08-01)

### Features

- Add initial versioning configuration and changelog documentation
  ([b134945](https://github.com/mkm29/stockula/commit/b134945a74a48d472e7859ef9a5879ff332a89c8))
- add pytest-xdist for parallel test execution and improve test d…
  ([e07846a](https://github.com/mkm29/stockula/commit/e07846a4a5ece26ad576fa5d81e2119bf4d059c7))
- add pytest-xdist for parallel test execution and improve test database handling
  ([006a288](https://github.com/mkm29/stockula/commit/006a288ebe50754418395f720a6cec4c7a1e1cbb))
- added pre-commit hooks to lint code, check data types
  ([4e577fb](https://github.com/mkm29/stockula/commit/4e577fb442e7ed24c618f8f8117b905d0d84e887))
- **config:** Add release configuration and manifest for version 0.2.…
  ([08486a4](https://github.com/mkm29/stockula/commit/08486a40bc9cbf5e593a56aa6f8bee8e89afcd7f))
- **config:** Add release configuration and manifest for version 0.2.0; update .gitignore to exclude JSON files
  ([7a21aff](https://github.com/mkm29/stockula/commit/7a21affad0efb0e4347452821f09c494265c709a))
- **docs:** Add development documentation for AutoTS threading considerations
  ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- **docs:** Update changelog and user guide for train/test evaluation enhancements
  ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- **docs:** Update changelog and user guide for train/test evaluation…
  ([82c2fbb](https://github.com/mkm29/stockula/commit/82c2fbbce7eb0ae10863ed3348e5125cd2c3f60e))
- **docs:** Update changelog for version 0.2.0; enhance forecast evaluation and backtest functionality
  ([e061515](https://github.com/mkm29/stockula/commit/e061515b033024d0ee2c3ac9e609a035b00b09eb))
- **domain:** Exclude tickers with 0% allocation from portfolio creation and tests
  ([ac987dc](https://github.com/mkm29/stockula/commit/ac987dc3dccd2ae83c5b9f6552dbca0e92a4af8e))
- Enhance main.py with portfolio summary and initial value calculations
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- **forecasting:** Implement train/test split evaluation for stock price forecasting
  ([ce74bd6](https://github.com/mkm29/stockula/commit/ce74bd64b610ac2e6b8dc1407f2dcc8864a77e24))
- Implement Ticker domain model with singleton registry
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Initialize SQLite database for stock data storage
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- **logging:** Add method to set module-specific logging levels
  ([8b95356](https://github.com/mkm29/stockula/commit/8b953561b30c73f58bd0d270e7b887f688a762e0))
- **logging:** Add method to set module-specific logging levels
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **logging:** Update LoggingManager to handle additional third-party libraries
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **main:** Add detailed portfolio holdings display in console output
  ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))
- **main:** Enhance forecasting output with return percentage and improve logging
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **main:** Enhance portfolio value display with forecasted values and accuracy metrics
  ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- update CI/CD workflows and documentation
  ([0e48c3d](https://github.com/mkm29/stockula/commit/0e48c3d4c4c637800e318f95276970ff611b9e8f))

### Bug Fixes

- **backtesting:** Suppress progress output in backtest runner
  ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- Correct import paths in main.py
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Display correct strategy-specific backtest results in portfolio summary
  ([6bd5a97](https://github.com/mkm29/stockula/commit/6bd5a9786e6662448d73229c387fe2c38ed1b788))
- **docs:** Update documentation links to use relative paths
  ([a5079ba](https://github.com/mkm29/stockula/commit/a5079bad4b6aaec35a8b19b3d8918ae0dd5e3686))
- update .gitignore and CHANGELOG for CLAUDE file renaming
  ([324c7e6](https://github.com/mkm29/stockula/commit/324c7e615841c48c713a03085515b7da490d79d6))
- update .gitignore and CHANGELOG for CLAUDE file renaming
  ([4fd14b1](https://github.com/mkm29/stockula/commit/4fd14b1cf4dde2500690d22ddb91fd8af0edc907))
- update README links to ensure proper navigation
  ([bd5e1df](https://github.com/mkm29/stockula/commit/bd5e1dff819f38d109ba041a84cd06b055bbf001))
- update README links to ensure proper navigation
  ([6563c19](https://github.com/mkm29/stockula/commit/6563c1949e250b21b964b264d47252457dc83d50))

### Documentation

- Add broker configuration documentation to README
  ([e474458](https://github.com/mkm29/stockula/commit/e47445876c2437c3fe5cea99054d587e308ea36b))
- Update README to include system requirements, dependencies, architecture overview, and data flow
  ([505b797](https://github.com/mkm29/stockula/commit/505b7973878f7e6fd541e12c79f02b9b61074aea))
- Update stockula.yml and stockula.yml.example with new structure
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Update storage requirement in README for data caching
  ([da020fa](https://github.com/mkm29/stockula/commit/da020fab7edd3e1de4a962f2620f85b4cf26f2a5))
- Update user guide and README to reflect sorting by return in analysis modes
  ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))

### Code Refactoring

- Create structured data models for backtest results
  ([d8b5fe2](https://github.com/mkm29/stockula/commit/d8b5fe2ff3a1ce63265fdf8afb5132e83a16adcf))
- Organize configuration files into examples directory
  ([ecc4c0b](https://github.com/mkm29/stockula/commit/ecc4c0bd62b57f47cbbe4fc5887f341cd12006ae))
- Update forecaster.py to streamline imports
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))

### Tests

- **autots:** Create direct test for AutoTS to isolate hanging issues
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add minimal test configuration for two tickers
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add quick test configuration with a single ticker
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add simple test configuration for forecast testing
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add test configuration for parallel progress tracking
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **parallel:** Implement tests for parallel forecasting to debug hanging issues
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **progress:** Create test script to demonstrate parallel forecasting progress tracking
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **sequential:** Implement sequential forecasting test to verify functionality
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **single:** Create test for single ticker forecasting to debug hanging issues
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **unit:** Refactor main tests to use parallel forecasting and improve logging checks
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **unit:** Update unit tests for forecaster to check output suppression
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))

### Miscellaneous Chores

- Add CLAUDE.market_data and .claude to .gitignore
  ([0cf0065](https://github.com/mkm29/stockula/commit/0cf006507c51b49bbd8b66b165c33e28434ccb81))
- Add example configuration files for different allocation strategies
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Add SQLite database initialization and testing script
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- comment out pull_request branch triggers in workflow files
  ([bc9ee6f](https://github.com/mkm29/stockula/commit/bc9ee6f54cc4c1ccdcd6826345b73cafdfa6a77f))
- copy in README.md in Dockerfile
  ([f6a8db0](https://github.com/mkm29/stockula/commit/f6a8db0f69652d56e8e562db6c44d8ae266a93cf))
- fixed pip install command in Dockerfile
  ([4765432](https://github.com/mkm29/stockula/commit/4765432a8cde9a789cbb16b8ea72fd2a42d408fc))
- **main:** release 0.2.0 ([dd76ab5](https://github.com/mkm29/stockula/commit/dd76ab5a102148ac6c6c2d80481b03ec348a1786))
- **main:** release 0.2.0 ([5fe8f0e](https://github.com/mkm29/stockula/commit/5fe8f0e9ddab552546f5843e8bd6695c782799d0))
- **main:** release stockula 0.2.0
  ([e779ca3](https://github.com/mkm29/stockula/commit/e779ca31ad6122c3320b419ce4766f851053e6c8))
- **main:** release stockula 0.2.0
  ([c7e6c33](https://github.com/mkm29/stockula/commit/c7e6c33a14c78307c234c23cc4ca52dc010811f2))
- **main:** release stockula 0.3.0
  ([45922f8](https://github.com/mkm29/stockula/commit/45922f84f7ca86183f8c6b1cb60d21913f3135f3))
- remove branch triggers for main in Docker and Release workflows
  ([c258b4e](https://github.com/mkm29/stockula/commit/c258b4ef1a8e41eaa8fb95d4fc7f40f8613be188))
- remove branch triggers for main in Docker and Release workflows
  ([7074d82](https://github.com/mkm29/stockula/commit/7074d82544f45156970e02dcc6dc7d5cf460283e))
- reset version to v0.1.0 for initial release
  ([036fa12](https://github.com/mkm29/stockula/commit/036fa12646f68a386c82b38dff7ec28766d344cd))
- Update .gitignore to include AutoTS cache and SQLite database files
  ([6117bd3](https://github.com/mkm29/stockula/commit/6117bd30f6d55fcf62b798eb2cf9ded29300cfb3))
- update changelog with test suite fixes and improvements; modif…
  ([cf734ea](https://github.com/mkm29/stockula/commit/cf734ea913c9819b78da6b5b2dcb6e1b4bedf44a))
- update changelog with test suite fixes and improvements; modify Dockerfile version; enhance testing documentation; add
  backward compatibility methods in DatabaseManager
  ([8555fc6](https://github.com/mkm29/stockula/commit/8555fc60e54affe6956cf18654b2b0f41352f315))
- updated .dockerignore in order to build uv project
  ([a4eb5da](https://github.com/mkm29/stockula/commit/a4eb5daf64c9bdcae3529f9864ac7adf35e96214))
- updated docker-build job to add build time labels
  ([71ecc8b](https://github.com/mkm29/stockula/commit/71ecc8b488e6b79b7550738894cf9826b0e4d90b))

## [0.2.0](https://github.com/mkm29/stockula/compare/v0.1.0...v0.2.0) (2025-07-31)

### Features

- Add initial versioning configuration and changelog documentation
  ([b134945](https://github.com/mkm29/stockula/commit/b134945a74a48d472e7859ef9a5879ff332a89c8))
- add pytest-xdist for parallel test execution and improve test d…
  ([e07846a](https://github.com/mkm29/stockula/commit/e07846a4a5ece26ad576fa5d81e2119bf4d059c7))
- add pytest-xdist for parallel test execution and improve test database handling
  ([006a288](https://github.com/mkm29/stockula/commit/006a288ebe50754418395f720a6cec4c7a1e1cbb))
- added pre-commit hooks to lint code, check data types
  ([4e577fb](https://github.com/mkm29/stockula/commit/4e577fb442e7ed24c618f8f8117b905d0d84e887))
- **config:** Add release configuration and manifest for version 0.2.…
  ([08486a4](https://github.com/mkm29/stockula/commit/08486a40bc9cbf5e593a56aa6f8bee8e89afcd7f))
- **config:** Add release configuration and manifest for version 0.2.0; update .gitignore to exclude JSON files
  ([7a21aff](https://github.com/mkm29/stockula/commit/7a21affad0efb0e4347452821f09c494265c709a))
- **docs:** Add development documentation for AutoTS threading considerations
  ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- **docs:** Update changelog and user guide for train/test evaluation enhancements
  ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- **docs:** Update changelog and user guide for train/test evaluation…
  ([82c2fbb](https://github.com/mkm29/stockula/commit/82c2fbbce7eb0ae10863ed3348e5125cd2c3f60e))
- **docs:** Update changelog for version 0.2.0; enhance forecast evaluation and backtest functionality
  ([e061515](https://github.com/mkm29/stockula/commit/e061515b033024d0ee2c3ac9e609a035b00b09eb))
- **domain:** Exclude tickers with 0% allocation from portfolio creation and tests
  ([ac987dc](https://github.com/mkm29/stockula/commit/ac987dc3dccd2ae83c5b9f6552dbca0e92a4af8e))
- Enhance main.py with portfolio summary and initial value calculations
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- **forecasting:** Implement train/test split evaluation for stock price forecasting
  ([ce74bd6](https://github.com/mkm29/stockula/commit/ce74bd64b610ac2e6b8dc1407f2dcc8864a77e24))
- Implement Ticker domain model with singleton registry
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Initialize SQLite database for stock data storage
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- **logging:** Add method to set module-specific logging levels
  ([8b95356](https://github.com/mkm29/stockula/commit/8b953561b30c73f58bd0d270e7b887f688a762e0))
- **logging:** Add method to set module-specific logging levels
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **logging:** Update LoggingManager to handle additional third-party libraries
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **main:** Add detailed portfolio holdings display in console output
  ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))
- **main:** Enhance forecasting output with return percentage and improve logging
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **main:** Enhance portfolio value display with forecasted values and accuracy metrics
  ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- update CI/CD workflows and documentation
  ([0e48c3d](https://github.com/mkm29/stockula/commit/0e48c3d4c4c637800e318f95276970ff611b9e8f))

### Bug Fixes

- **backtesting:** Suppress progress output in backtest runner
  ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
- Correct import paths in main.py
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Display correct strategy-specific backtest results in portfolio summary
  ([6bd5a97](https://github.com/mkm29/stockula/commit/6bd5a9786e6662448d73229c387fe2c38ed1b788))
- **docs:** Update documentation links to use relative paths
  ([a5079ba](https://github.com/mkm29/stockula/commit/a5079bad4b6aaec35a8b19b3d8918ae0dd5e3686))
- update .gitignore and CHANGELOG for CLAUDE file renaming
  ([324c7e6](https://github.com/mkm29/stockula/commit/324c7e615841c48c713a03085515b7da490d79d6))
- update .gitignore and CHANGELOG for CLAUDE file renaming
  ([4fd14b1](https://github.com/mkm29/stockula/commit/4fd14b1cf4dde2500690d22ddb91fd8af0edc907))

### Documentation

- Add broker configuration documentation to README
  ([e474458](https://github.com/mkm29/stockula/commit/e47445876c2437c3fe5cea99054d587e308ea36b))
- Update README to include system requirements, dependencies, architecture overview, and data flow
  ([505b797](https://github.com/mkm29/stockula/commit/505b7973878f7e6fd541e12c79f02b9b61074aea))
- Update stockula.yml and stockula.yml.example with new structure
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Update storage requirement in README for data caching
  ([da020fa](https://github.com/mkm29/stockula/commit/da020fab7edd3e1de4a962f2620f85b4cf26f2a5))
- Update user guide and README to reflect sorting by return in analysis modes
  ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))

### Code Refactoring

- Create structured data models for backtest results
  ([d8b5fe2](https://github.com/mkm29/stockula/commit/d8b5fe2ff3a1ce63265fdf8afb5132e83a16adcf))
- Organize configuration files into examples directory
  ([ecc4c0b](https://github.com/mkm29/stockula/commit/ecc4c0bd62b57f47cbbe4fc5887f341cd12006ae))
- Update forecaster.py to streamline imports
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))

### Tests

- **autots:** Create direct test for AutoTS to isolate hanging issues
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add minimal test configuration for two tickers
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add quick test configuration with a single ticker
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add simple test configuration for forecast testing
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **config:** Add test configuration for parallel progress tracking
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **parallel:** Implement tests for parallel forecasting to debug hanging issues
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **progress:** Create test script to demonstrate parallel forecasting progress tracking
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **sequential:** Implement sequential forecasting test to verify functionality
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **single:** Create test for single ticker forecasting to debug hanging issues
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **unit:** Refactor main tests to use parallel forecasting and improve logging checks
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
- **unit:** Update unit tests for forecaster to check output suppression
  ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))

### Miscellaneous Chores

- Add CLAUDE.market_data and .claude to .gitignore
  ([0cf0065](https://github.com/mkm29/stockula/commit/0cf006507c51b49bbd8b66b165c33e28434ccb81))
- Add example configuration files for different allocation strategies
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- Add SQLite database initialization and testing script
  ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
- comment out pull_request branch triggers in workflow files
  ([bc9ee6f](https://github.com/mkm29/stockula/commit/bc9ee6f54cc4c1ccdcd6826345b73cafdfa6a77f))
- copy in README.md in Dockerfile
  ([f6a8db0](https://github.com/mkm29/stockula/commit/f6a8db0f69652d56e8e562db6c44d8ae266a93cf))
- fixed pip install command in Dockerfile
  ([4765432](https://github.com/mkm29/stockula/commit/4765432a8cde9a789cbb16b8ea72fd2a42d408fc))
- **main:** release stockula 0.2.0
  ([e779ca3](https://github.com/mkm29/stockula/commit/e779ca31ad6122c3320b419ce4766f851053e6c8))
- **main:** release stockula 0.2.0
  ([c7e6c33](https://github.com/mkm29/stockula/commit/c7e6c33a14c78307c234c23cc4ca52dc010811f2))
- **main:** release stockula 0.3.0
  ([45922f8](https://github.com/mkm29/stockula/commit/45922f84f7ca86183f8c6b1cb60d21913f3135f3))
- remove branch triggers for main in Docker and Release workflows
  ([c258b4e](https://github.com/mkm29/stockula/commit/c258b4ef1a8e41eaa8fb95d4fc7f40f8613be188))
- remove branch triggers for main in Docker and Release workflows
  ([7074d82](https://github.com/mkm29/stockula/commit/7074d82544f45156970e02dcc6dc7d5cf460283e))
- reset version to v0.1.0 for initial release
  ([036fa12](https://github.com/mkm29/stockula/commit/036fa12646f68a386c82b38dff7ec28766d344cd))
- Update .gitignore to include AutoTS cache and SQLite database files
  ([6117bd3](https://github.com/mkm29/stockula/commit/6117bd30f6d55fcf62b798eb2cf9ded29300cfb3))
- update changelog with test suite fixes and improvements; modif…
  ([cf734ea](https://github.com/mkm29/stockula/commit/cf734ea913c9819b78da6b5b2dcb6e1b4bedf44a))
- update changelog with test suite fixes and improvements; modify Dockerfile version; enhance testing documentation; add
  backward compatibility methods in DatabaseManager
  ([8555fc6](https://github.com/mkm29/stockula/commit/8555fc60e54affe6956cf18654b2b0f41352f315))
- updated .dockerignore in order to build uv project
  ([a4eb5da](https://github.com/mkm29/stockula/commit/a4eb5daf64c9bdcae3529f9864ac7adf35e96214))
- updated docker-build job to add build time labels
  ([71ecc8b](https://github.com/mkm29/stockula/commit/71ecc8b488e6b79b7550738894cf9826b0e4d90b))

## Changelog

All notable changes to this project will be documented in this file.

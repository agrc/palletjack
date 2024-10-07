# Changelog

## [5.1.1-0](https://github.com/agrc/palletjack/compare/v5.1.0...v5.1.1-0) (2024-10-07)


### Bug Fixes

* .sr has attributes, not keys ([3004900](https://github.com/agrc/palletjack/commit/3004900de9b23c24172373770ce53acac59552c4))
* make log message more generic (and correct) ([270e4ba](https://github.com/agrc/palletjack/commit/270e4bacf0f2bef01b96891cd98b1d4488f94e86))
* remove date parsing from PostgresLoader ([cc5ca85](https://github.com/agrc/palletjack/commit/cc5ca85ed3d608eb5c8d5083c5bb039a96d4596a))


### Dependencies

* update arcgis requirement in the major-dependencies group ([66316c1](https://github.com/agrc/palletjack/commit/66316c1875c438fe84d502f48c56cac6da607f48))

## [5.1.0](https://github.com/agrc/palletjack/compare/v5.0.2...v5.1.0) (2024-10-04)


### Features

* enhance cloud check in PostgresLoader to work in any GCP environment ([05748f2](https://github.com/agrc/palletjack/commit/05748f20cf203bd2cfa30143fa7caaaaea947152))

## [5.0.2](https://github.com/agrc/palletjack/compare/v5.0.1...v5.0.2) (2024-09-16)


### Dependencies

* update geodatasets requirement in the major-dependencies group ([5b11f93](https://github.com/agrc/palletjack/commit/5b11f93535c8503ed1afd8fc807a5df3189191cb))

## [5.0.1](https://github.com/agrc/palletjack/compare/v5.0.0...v5.0.1) (2024-08-16)


### Bug Fixes

* date value in log to match text around it ([c09702b](https://github.com/agrc/palletjack/commit/c09702b836cea219ffaf84bf1c7533f09e3f8bf2))

## [5.0.0](https://github.com/agrc/palletjack/compare/4.4.2...v5.0.0) (2024-08-15)


### ⚠ BREAKING CHANGES

* FeatureServiceUpdater -> ServiceUpdater

### Features

* implement ServiceUpdater ([c538572](https://github.com/agrc/palletjack/commit/c5385721256d0bf1b7f854fb9251192dd67c9df3)), closes [#92](https://github.com/agrc/palletjack/issues/92)


### Bug Fixes

* remove version number in comments that is confusion release-please ([f5d9647](https://github.com/agrc/palletjack/commit/f5d96472076bb21f32542bc56ad2a2f5e9859dc6))
* run ruff as separate command ([4dc1361](https://github.com/agrc/palletjack/commit/4dc1361c8de9f64403f4512180a12262ad812e33))
* set python version to help with action caching ([b03756d](https://github.com/agrc/palletjack/commit/b03756dfdf197d96a14697ca4dce668472757336))


### Dependencies

* update geodatasets requirement ([630f07b](https://github.com/agrc/palletjack/commit/630f07b8beb9519da38c874fbd9c10b44ca20b73))

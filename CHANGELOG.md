# Changelog

## [5.2.11](https://github.com/agrc/palletjack/compare/v5.2.10...v5.2.11) (2025-12-05)


### Bug Fixes

* empty commit for version bump ([a63ff83](https://github.com/agrc/palletjack/commit/a63ff83b484bed73e534c9fcebd1a515cf2399de))

## [5.2.10](https://github.com/agrc/palletjack/compare/v5.2.9...v5.2.10) (2025-12-05)


### Bug Fixes

* continue if it can't find temp gdb to delete ([a0ea05e](https://github.com/agrc/palletjack/commit/a0ea05e8815d8018496d0950a035bc62d3fad6c4))


### Dependencies

* preemptive dbot ([b69ef8d](https://github.com/agrc/palletjack/commit/b69ef8d0eecb5a1ef41f0480520f20a78b08f202))

## [5.2.10-rc.1](https://github.com/agrc/palletjack/compare/v5.2.9...v5.2.10-rc.1) (2025-12-05)


### Bug Fixes

* continue if it can't find temp gdb to delete ([de67c91](https://github.com/agrc/palletjack/commit/de67c91e7ed86787a9024395fb157887e7fd9ed6))

## [5.2.9](https://github.com/agrc/palletjack/compare/v5.2.8...v5.2.9) (2025-11-21)


### Dependencies

* q4 dbot, dbot frequency ([543b32a](https://github.com/agrc/palletjack/commit/543b32aaa8edf9f4d61597245689b39456fb7788))

## [5.2.8](https://github.com/agrc/palletjack/compare/v5.2.7...v5.2.8) (2025-09-18)


### Bug Fixes

* handle promote_to_multi check failing on non-spatial gdfs ([b4600cc](https://github.com/agrc/palletjack/commit/b4600cc19f54cbf4cfcddabc5203547cea5bbfe4))

## [5.2.7](https://github.com/agrc/palletjack/compare/v5.2.6...v5.2.7) (2025-09-02)


### Features

* separate gdb upload from service update ([64b98f7](https://github.com/agrc/palletjack/commit/64b98f75c4edfc5238563cfb15348b2b55ea260b))

## [5.2.6](https://github.com/agrc/palletjack/compare/v5.2.5...v5.2.6) (2025-08-19)


### Bug Fixes

* don't condense points and multipoints in field check ([fca734c](https://github.com/agrc/palletjack/commit/fca734ce0aebed373edc346f84d9eb5999aa7b87))
* don't promote points to multipoints ([8dbe8c1](https://github.com/agrc/palletjack/commit/8dbe8c11ea98b49bc731fcac903dc3cda55f4ae9))

## [5.2.5](https://github.com/agrc/palletjack/compare/v5.2.4...v5.2.5) (2025-08-06)


### Bug Fixes

* **deps:** pin to working version of paramiko ([71574bf](https://github.com/agrc/palletjack/commit/71574bf2b10078a91e027025141a0c3d275efecf))

## [5.2.4](https://github.com/agrc/palletjack/compare/v5.2.3...v5.2.4) (2025-07-31)


### Features

* generalized method to convert any df to gdf ([014fe39](https://github.com/agrc/palletjack/commit/014fe39d5a5f8f7575951d5db7dafc9468db2558))


### Bug Fixes

* handle gdf types in geometry checker ([e8591a4](https://github.com/agrc/palletjack/commit/e8591a4a87cb5ece89fc963519570a2242e77f7b))


### Dependencies

* update for dbot ([a5765bc](https://github.com/agrc/palletjack/commit/a5765bc2294b5d7e525d887f25e6131ad9e0dee8))

## [5.2.3](https://github.com/agrc/palletjack/compare/v5.2.2...v5.2.3) (2025-06-23)


### Bug Fixes

* delete any existing gdb upload items left over from previous runs ([11f3359](https://github.com/agrc/palletjack/commit/11f33593c412172b6ee4be6c777e361c798d3236)), closes [#78](https://github.com/agrc/palletjack/issues/78)

## [5.2.2](https://github.com/agrc/palletjack/compare/v5.2.1...v5.2.2) (2025-05-19)


### Bug Fixes

* prevent FieldChecker from returning false negatives related to null string values ([b1f74a5](https://github.com/agrc/palletjack/commit/b1f74a5e977235d24730d59841cf76b3089b5df0))

## [5.2.1](https://github.com/agrc/palletjack/compare/v5.2.0...v5.2.1) (2025-02-20)


### Bug Fixes

* switch to post for query request to accommodate larger where clauses ([70643d4](https://github.com/agrc/palletjack/commit/70643d4c035eba920df866f0ca16fd4a9077aaef))

## [5.2.0](https://github.com/agrc/palletjack/compare/v5.1.2...v5.2.0) (2025-02-20)


### Features

* add token support to RESTServiceLoader ([867abaa](https://github.com/agrc/palletjack/commit/867abaa8f2eae6e7a682edeb7233c61da8b782b2))

## [5.1.2](https://github.com/agrc/palletjack/compare/v5.1.1...v5.1.2) (2025-01-15)


### Dependencies

* **dev:** update pytest-cov requirement from &lt;6,&gt;=3 to &gt;=3,&lt;7 ([dfd19e1](https://github.com/agrc/palletjack/commit/dfd19e1e6aad0ac9a4f5f0b82d6540d9b0953c76))

## [5.1.1](https://github.com/agrc/palletjack/compare/v5.1.0...v5.1.1) (2024-10-08)


### Bug Fixes

* .sr has attributes, not keys ([09a3c0b](https://github.com/agrc/palletjack/commit/09a3c0b78227912e4a3b7bde4afcce082d584ee3))
* access sr as attr, not dict ([5c7dbb2](https://github.com/agrc/palletjack/commit/5c7dbb2f8b2204f0c5d4006b087bfd7c5b86c053))
* make log message more generic (and correct) ([85314da](https://github.com/agrc/palletjack/commit/85314dad4fe7994d0be6113c10637b9b6010edba))
* remove date parsing from PostgresLoader ([fc431ab](https://github.com/agrc/palletjack/commit/fc431ab4c5875ab606cd82a008df21e4ba7e849e))


### Dependencies

* update arcgis requirement in the major-dependencies group ([5825933](https://github.com/agrc/palletjack/commit/5825933da30c1f7150834181014d050222383e29))

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

/**
 * Lazy-loaded page components for React Router.
 *
 * Each page is loaded on demand via React.lazy() and Suspense,
 * enabling code splitting per route.
 */
import { lazy } from 'react'

export const LandingPage = lazy(() => import('./LandingPage'))
export const BatchReviewPage = lazy(() => import('./BatchReviewPage'))
export const OperatorPage = lazy(() => import('./OperatorPage'))
export const InferencePage = lazy(() => import('./InferencePage'))
export const HarmonicsPage = lazy(() => import('./HarmonicsPage'))
export const ExperimentPage = lazy(() => import('./ExperimentPage'))
export const DatasetExplorerPage = lazy(() => import('./DatasetExplorerPage'))
export const DocumentRetrievalPage = lazy(() => import('./DocumentRetrievalPage'))
export const DevelopmentPage = lazy(() => import('./DevelopmentPage'))
export const GraphPage = lazy(() => import('./GraphPage'))
export const SinditPage = lazy(() => import('./SinditPage'))
export const LearningsPage = lazy(() => import('./LearningsPage'))
export const SettingsPage = lazy(() => import('./SettingsPage'))

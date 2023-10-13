/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef _itkKNNGraphAlphaMutualInformationImageToImageMetric_hxx
#define _itkKNNGraphAlphaMutualInformationImageToImageMetric_hxx

#include "itkKNNGraphAlphaMutualInformationImageToImageMetric.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template< class TFixedImage, class TMovingImage >
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::KNNGraphAlphaMutualInformationImageToImageMetric()
{
  this->SetComputeGradient( false ); // don't use the default gradient
  this->SetUseImageSampler( true );
  this->m_Alpha           = 0.99;
  this->m_AvoidDivisionBy = 1e-10;

  this->m_BinaryKNNTreeFixed  = 0;
  this->m_BinaryKNNTreeMoving = 0;
  this->m_BinaryKNNTreeJoint  = 0;

  this->m_BinaryKNNTreeSearcherFixed  = 0;
  this->m_BinaryKNNTreeSearcherMoving = 0;
  this->m_BinaryKNNTreeSearcherJoint  = 0;

  this->m_UseKNNMultiThread = false;
  this->m_KNNNumberOfThreads = 2;

  /** Initialize the m_KNNThreaderMetricParameters. */
  this->m_KNNThreaderMetricParameters.st_Metric = this;

  // Multi-threading structs
  this->m_KNNGetValueAndDerivativePerThreadVariables     = NULL;
  this->m_KNNGetValueAndDerivativePerThreadVariablesSize = 0;

} // end Constructor()


/**
 * ********************* Destructor ****************************
 */

template< class TFixedImage, class TMovingImage >
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::~KNNGraphAlphaMutualInformationImageToImageMetric()
{
  delete[] this->m_KNNGetValueAndDerivativePerThreadVariables;

} // end Destructor


/**
 * ************************ SetANNkDTree *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::SetANNkDTree( unsigned int bucketSize, std::string splittingRule )
{
  this->SetANNkDTree( bucketSize, splittingRule, splittingRule, splittingRule );

} // end SetANNkDTree()


/**
 * ************************ SetANNkDTree *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::SetANNkDTree( unsigned int bucketSize,
  std::string splittingRuleFixed,
  std::string splittingRuleMoving,
  std::string splittingRuleJoint )
{
  typename ANNkDTreeType::Pointer tmpPtrF = ANNkDTreeType::New();
  typename ANNkDTreeType::Pointer tmpPtrM = ANNkDTreeType::New();
  typename ANNkDTreeType::Pointer tmpPtrJ = ANNkDTreeType::New();

  tmpPtrF->SetBucketSize( bucketSize );
  tmpPtrM->SetBucketSize( bucketSize );
  tmpPtrJ->SetBucketSize( bucketSize );

  tmpPtrF->SetSplittingRule( splittingRuleFixed );
  tmpPtrM->SetSplittingRule( splittingRuleMoving );
  tmpPtrJ->SetSplittingRule( splittingRuleJoint );

  this->m_BinaryKNNTreeFixed  = tmpPtrF;
  this->m_BinaryKNNTreeMoving = tmpPtrM;
  this->m_BinaryKNNTreeJoint  = tmpPtrJ;

} // end SetANNkDTree()


/**
 * ************************ SetANNbdTree *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::SetANNbdTree( unsigned int bucketSize,
  std::string splittingRule,
  std::string shrinkingRule )
{
  this->SetANNbdTree( bucketSize,
    splittingRule, splittingRule, splittingRule,
    shrinkingRule, shrinkingRule, shrinkingRule );

} // end SetANNbdTree()


/**
 * ************************ SetANNbdTree *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::SetANNbdTree( unsigned int bucketSize, std::string splittingRuleFixed,
  std::string splittingRuleMoving, std::string splittingRuleJoint,
  std::string shrinkingRuleFixed, std::string shrinkingRuleMoving,
  std::string shrinkingRuleJoint )
{
  typename ANNbdTreeType::Pointer tmpPtrF = ANNbdTreeType::New();
  typename ANNbdTreeType::Pointer tmpPtrM = ANNbdTreeType::New();
  typename ANNbdTreeType::Pointer tmpPtrJ = ANNbdTreeType::New();

  tmpPtrF->SetBucketSize( bucketSize );
  tmpPtrM->SetBucketSize( bucketSize );
  tmpPtrJ->SetBucketSize( bucketSize );

  tmpPtrF->SetSplittingRule( splittingRuleFixed );
  tmpPtrM->SetSplittingRule( splittingRuleMoving );
  tmpPtrJ->SetSplittingRule( splittingRuleJoint );

  tmpPtrF->SetShrinkingRule( shrinkingRuleFixed );
  tmpPtrM->SetShrinkingRule( shrinkingRuleMoving );
  tmpPtrJ->SetShrinkingRule( shrinkingRuleJoint );

  this->m_BinaryKNNTreeFixed  = tmpPtrF;
  this->m_BinaryKNNTreeMoving = tmpPtrM;
  this->m_BinaryKNNTreeJoint  = tmpPtrJ;

} // end SetANNbdTree()


/**
 * ************************ SetANNBruteForceTree *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::SetANNBruteForceTree( void )
{
  this->m_BinaryKNNTreeFixed  = ANNBruteForceTreeType::New();
  this->m_BinaryKNNTreeMoving = ANNBruteForceTreeType::New();
  this->m_BinaryKNNTreeJoint  = ANNBruteForceTreeType::New();

} // end SetANNBruteForceTree()


/**
 * ************************ SetANNStandardTreeSearch *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::SetANNStandardTreeSearch(
  unsigned int kNearestNeighbors,
  double errorBound )
{
  typename ANNStandardTreeSearchType::Pointer tmpPtrF
    = ANNStandardTreeSearchType::New();
  typename ANNStandardTreeSearchType::Pointer tmpPtrM
    = ANNStandardTreeSearchType::New();
  typename ANNStandardTreeSearchType::Pointer tmpPtrJ
    = ANNStandardTreeSearchType::New();

  tmpPtrF->SetKNearestNeighbors( kNearestNeighbors );
  tmpPtrM->SetKNearestNeighbors( kNearestNeighbors );
  tmpPtrJ->SetKNearestNeighbors( kNearestNeighbors );

  tmpPtrF->SetErrorBound( errorBound );
  tmpPtrM->SetErrorBound( errorBound );
  tmpPtrJ->SetErrorBound( errorBound );

  this->m_BinaryKNNTreeSearcherFixed  = tmpPtrF;
  this->m_BinaryKNNTreeSearcherMoving = tmpPtrM;
  this->m_BinaryKNNTreeSearcherJoint  = tmpPtrJ;

} // end SetANNStandardTreeSearch()


/**
 * ************************ SetANNFixedRadiusTreeSearch *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::SetANNFixedRadiusTreeSearch(
  unsigned int kNearestNeighbors,
  double errorBound,
  double squaredRadius )
{
  typename ANNFixedRadiusTreeSearchType::Pointer tmpPtrF
    = ANNFixedRadiusTreeSearchType::New();
  typename ANNFixedRadiusTreeSearchType::Pointer tmpPtrM
    = ANNFixedRadiusTreeSearchType::New();
  typename ANNFixedRadiusTreeSearchType::Pointer tmpPtrJ
    = ANNFixedRadiusTreeSearchType::New();

  tmpPtrF->SetKNearestNeighbors( kNearestNeighbors );
  tmpPtrM->SetKNearestNeighbors( kNearestNeighbors );
  tmpPtrJ->SetKNearestNeighbors( kNearestNeighbors );

  tmpPtrF->SetErrorBound( errorBound );
  tmpPtrM->SetErrorBound( errorBound );
  tmpPtrJ->SetErrorBound( errorBound );

  tmpPtrF->SetSquaredRadius( squaredRadius );
  tmpPtrM->SetSquaredRadius( squaredRadius );
  tmpPtrJ->SetSquaredRadius( squaredRadius );

  this->m_BinaryKNNTreeSearcherFixed  = tmpPtrF;
  this->m_BinaryKNNTreeSearcherMoving = tmpPtrM;
  this->m_BinaryKNNTreeSearcherJoint  = tmpPtrJ;

} // end SetANNFixedRadiusTreeSearch()


/**
 * ************************ SetANNPriorityTreeSearch *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::SetANNPriorityTreeSearch(
  unsigned int kNearestNeighbors,
  double errorBound )
{
  typename ANNPriorityTreeSearchType::Pointer tmpPtrF
    = ANNPriorityTreeSearchType::New();
  typename ANNPriorityTreeSearchType::Pointer tmpPtrM
    = ANNPriorityTreeSearchType::New();
  typename ANNPriorityTreeSearchType::Pointer tmpPtrJ
    = ANNPriorityTreeSearchType::New();

  tmpPtrF->SetKNearestNeighbors( kNearestNeighbors );
  tmpPtrM->SetKNearestNeighbors( kNearestNeighbors );
  tmpPtrJ->SetKNearestNeighbors( kNearestNeighbors );

  tmpPtrF->SetErrorBound( errorBound );
  tmpPtrM->SetErrorBound( errorBound );
  tmpPtrJ->SetErrorBound( errorBound );

  this->m_BinaryKNNTreeSearcherFixed  = tmpPtrF;
  this->m_BinaryKNNTreeSearcherMoving = tmpPtrM;
  this->m_BinaryKNNTreeSearcherJoint  = tmpPtrJ;

} // end SetANNPriorityTreeSearch()


/**
 * ********************* Initialize *****************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::Initialize( void ) throw ( ExceptionObject )
{
  /** Call the superclass. */
  this->Superclass::Initialize();

  /** Check if the kNN trees are set. We only need to check the fixed tree. */
  if( !this->m_BinaryKNNTreeFixed )
  {
    itkExceptionMacro( << "ERROR: The kNN tree is not set. " );
  }

  /** Check if the kNN tree searchers are set. We only need to check the fixed searcher. */
  if( !this->m_BinaryKNNTreeSearcherFixed )
  {
    itkExceptionMacro( << "ERROR: The kNN tree searcher is not set. " );
  }

  /** Create the multi-threading handle. */
  if( this->m_UseKNNMultiThread )
	this->m_KNNThreader = KNNThreaderType::New();

} // end Initialize()


/**
 * ********************* InitializeKNNThreadingParameters ****************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::InitializeKNNThreadingParameters( void ) const
{
	/** Resize and initialize the threading related parameters.
	* The SetSize() functions do not resize the data when this is not
	* needed, which saves valuable re-allocation time.
	*
	* This function is only to be called at the start of each resolution.
	* Re-initialization of the potentially large vectors is performed after
	* each iteration, in the accumulate functions, in a multi-threaded fashion.
	* This has performance benefits for larger vector sizes.
	*/

	/** Only resize the array of structs when needed. */
	if( this->m_KNNGetValueAndDerivativePerThreadVariablesSize != this->m_KNNNumberOfThreads )
	{
	  delete[] this->m_KNNGetValueAndDerivativePerThreadVariables;
	  this->m_KNNGetValueAndDerivativePerThreadVariables     = new AlignedKNNGetValueAndDerivativePerThreadStruct[ this->m_KNNNumberOfThreads ];
	  this->m_KNNGetValueAndDerivativePerThreadVariablesSize = this->m_KNNNumberOfThreads;
	}

	/** Some initialization. */	
	this->m_KNNThreaderMetricParameters.st_FixedListSample = ListSampleType::New();
	this->m_KNNThreaderMetricParameters.st_MovingListSample = ListSampleType::New();
	this->m_KNNThreaderMetricParameters.st_JointListSample = ListSampleType::New();
	this->m_KNNThreaderMetricParameters.st_KNNContrib.SetSize(this->GetNumberOfParameters());
	this->m_KNNThreaderMetricParameters.st_KNNContrib.Fill(NumericTraits< DerivativeValueType >::ZeroValue());

	for( ThreadIdType i = 0; i < this->m_KNNNumberOfThreads; ++i )
	{
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_fixedListSample   = ListSampleType::New();
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_movingListSample  = ListSampleType::New();
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_jointListSample   = ListSampleType::New();

	  this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_sumG = NumericTraits< MeasureType >::Zero;
      this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_contribution.SetSize( this->GetNumberOfParameters() );
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_contribution.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
	}  

} // end InitializeKNNThreadingParameters()


/**
 * ************************ GetValue *************************
 */

template< class TFixedImage, class TMovingImage >
typename KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >::MeasureType
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::GetValue( const TransformParametersType & parameters ) const
{
  /** Initialize some variables. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /**
   * *************** Create the three list samples ******************
   */

  /** Create list samples. */
  ListSamplePointer listSampleFixed  = ListSampleType::New();
  ListSamplePointer listSampleMoving = ListSampleType::New();
  ListSamplePointer listSampleJoint  = ListSampleType::New();

  /** Compute the three list samples. */
  TransformJacobianContainerType        dummyJacobianContainer;
  TransformJacobianIndicesContainerType dummyJacobianIndicesContainer;
  SpatialDerivativeContainerType        dummySpatialDerivativesContainer;
  this->ComputeListSampleValuesAndDerivativePlusJacobian(
    listSampleFixed, listSampleMoving, listSampleJoint,
    false, dummyJacobianContainer, dummyJacobianIndicesContainer,
    dummySpatialDerivativesContainer );

  /** Check if enough samples were valid. */
  unsigned long size = this->GetImageSampler()->GetOutput()->Size();
  this->CheckNumberOfSamples( size,
    this->m_NumberOfPixelsCounted );

  /**
   * *************** Generate the three trees ******************
   *
   * and connect them to the searchers.
   */

  /** Generate the tree for the fixed image samples. */
  this->m_BinaryKNNTreeFixed->SetSample( listSampleFixed );
  this->m_BinaryKNNTreeFixed->GenerateTree();

  /** Generate the tree for the moving image samples. */
  this->m_BinaryKNNTreeMoving->SetSample( listSampleMoving );
  this->m_BinaryKNNTreeMoving->GenerateTree();

  /** Generate the tree for the joint image samples. */
  this->m_BinaryKNNTreeJoint->SetSample( listSampleJoint );
  this->m_BinaryKNNTreeJoint->GenerateTree();

  /** Initialize tree searchers. */
  this->m_BinaryKNNTreeSearcherFixed
  ->SetBinaryTree( this->m_BinaryKNNTreeFixed );
  this->m_BinaryKNNTreeSearcherMoving
  ->SetBinaryTree( this->m_BinaryKNNTreeMoving );
  this->m_BinaryKNNTreeSearcherJoint
  ->SetBinaryTree( this->m_BinaryKNNTreeJoint );

  /**
   * *************** Estimate the \alpha MI ******************
   *
   * This is done by searching for the nearest neighbours of each point
   * and calculating the distances.
   *
   * The estimate for the alpha - mutual information is given by:
   *
   *  \alpha MI = 1 / ( \alpha - 1 ) * \log 1/n^\alpha * \sum_{i=1}^n \sum_{p=1}^k
   *              ( jointLength / \sqrt( fixedLength * movingLength ) )^(2 \gamma),
   *
   * where
   *   - \alpha is set by the user and refers to \alpha - mutual information
   *   - n is the number of samples
   *   - k is the number of nearest neighbours
   *   - jointLength  is the distances to one of the nearest neighbours in listSampleJoint
   *   - fixedLength  is the distances to one of the nearest neighbours in listSampleFixed
   *   - movingLength is the distances to one of the nearest neighbours in listSampleMoving
   *   - \gamma relates to the distance metric and relates to \alpha as:
   *
   *        \gamma = d * ( 1 - \alpha ),
   *
   *     where d is the dimension of the feature space.
   *
   * In the original paper it is assumed that the mutual information of
   * two feature sets of equal dimension is calculated. If this is not
   * true, then
   *
   *        \gamma = ( ( d1 + d2 ) / 2 ) * ( 1 - alpha ),
   *
   * where d1 and d2 are the possibly different dimensions of the two feature sets.
   */

  /** Temporary variables. */
  typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
  MeasurementVectorType z_F, z_M, z_J;
  IndexArrayType        indices_F, indices_M, indices_J;
  DistanceArrayType     distances_F, distances_M, distances_J;

  MeasureType    H, G;
  AccumulateType sumG = NumericTraits< AccumulateType >::Zero;

  /** Get the size of the feature vectors. */
  unsigned int fixedSize  = this->GetNumberOfFixedImages();
  unsigned int movingSize = this->GetNumberOfMovingImages();
  unsigned int jointSize  = fixedSize + movingSize;

  /** Get the number of neighbours and \gamma. */
  unsigned int k        = this->m_BinaryKNNTreeSearcherFixed->GetKNearestNeighbors();
  double       twoGamma = jointSize * ( 1.0 - this->m_Alpha );

  /** Loop over all query points, i.e. all samples. */
  for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
  {
    /** Get the i-th query point. */
    listSampleFixed->GetMeasurementVector(  i, z_F );
    listSampleMoving->GetMeasurementVector( i, z_M );
    listSampleJoint->GetMeasurementVector(  i, z_J );

    /** Search for the K nearest neighbours of the current query point. */
    this->m_BinaryKNNTreeSearcherFixed->Search(  z_F, indices_F, distances_F );
    this->m_BinaryKNNTreeSearcherMoving->Search( z_M, indices_M, distances_M );
    this->m_BinaryKNNTreeSearcherJoint->Search(  z_J, indices_J, distances_J );

    /** Add the distances between the points to get the total graph length.
     * The outcommented implementation calculates: sum J/sqrt(F*M)
     *
    for ( unsigned int j = 0; j < K; j++ )
    {
    enumerator = vcl_sqrt( distsJ[ j ] );
    denominator = vcl_sqrt( vcl_sqrt( distsF[ j ] ) * vcl_sqrt( distsM[ j ] ) );
    if ( denominator > 1e-14 )
    {
    contribution += vcl_pow( enumerator / denominator, twoGamma );
    }
    }*/

    /** Add the distances of all neighbours of the query point,
    * for the three graphs:
    * sum M / sqrt( sum F * sum M)
    */

    /** Variables to compute the measure. */
    AccumulateType Gamma_F = NumericTraits< AccumulateType >::Zero;
    AccumulateType Gamma_M = NumericTraits< AccumulateType >::Zero;
    AccumulateType Gamma_J = NumericTraits< AccumulateType >::Zero;

    /** Loop over the neighbours. */
    for( unsigned int p = 0; p < k; p++ )
    {
      Gamma_F += vcl_sqrt( distances_F[ p ] );
      Gamma_M += vcl_sqrt( distances_M[ p ] );
      Gamma_J += vcl_sqrt( distances_J[ p ] );
    } // end loop over the k neighbours

    /** Calculate the contribution of this query point. */
    H = vcl_sqrt( Gamma_F * Gamma_M );
    if( H > this->m_AvoidDivisionBy )
    {
      /** Compute some sums. */
      G     = Gamma_J / H;
      sumG += vcl_pow( G, twoGamma );
    }
  } // end looping over all query points

  /**
   * *************** Finally, calculate the metric value \alpha MI ******************
   */

  double n, number;
  if( sumG > this->m_AvoidDivisionBy )
  {
    /** Compute the measure. */
    n       = static_cast< double >( this->m_NumberOfPixelsCounted );
    number  = vcl_pow( n, this->m_Alpha );
    measure = vcl_log( sumG / number ) / ( this->m_Alpha - 1.0 );
  }

  /** Return the negative alpha - mutual information. */
  return -measure;

} // end GetValue()


/**
 * ************************ GetDerivative *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ************************ GetValueAndDerivativeSingleThreaded *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivativeSingleThreaded(
  const TransformParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** Initialize some variables. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /**
   * *************** Create the three list samples ******************
   */

  /** Create list samples. */
  ListSamplePointer listSampleFixed  = ListSampleType::New();
  ListSamplePointer listSampleMoving = ListSampleType::New();
  ListSamplePointer listSampleJoint  = ListSampleType::New();

  /** Compute the three list samples and the derivatives. */
  TransformJacobianContainerType        jacobianContainer;
  TransformJacobianIndicesContainerType jacobianIndicesContainer;
  SpatialDerivativeContainerType        spatialDerivativesContainer;
  this->ComputeListSampleValuesAndDerivativePlusJacobian(
	listSampleFixed, listSampleMoving, listSampleJoint,
	true, jacobianContainer, jacobianIndicesContainer, spatialDerivativesContainer );

  /** Check if enough samples were valid. */
  unsigned long size = this->GetImageSampler()->GetOutput()->Size();
  this->CheckNumberOfSamples( size, this->m_NumberOfPixelsCounted );

  /**
   * *************** Generate the three trees ******************
   *
   * and connect them to the searchers.
   */

  /** Generate the tree for the fixed image samples. */
  this->m_BinaryKNNTreeFixed->SetSample( listSampleFixed );
  this->m_BinaryKNNTreeFixed->GenerateTree();

  /** Generate the tree for the moving image samples. */
  this->m_BinaryKNNTreeMoving->SetSample( listSampleMoving );
  this->m_BinaryKNNTreeMoving->GenerateTree();

  /** Generate the tree for the joint image samples. */
  this->m_BinaryKNNTreeJoint->SetSample( listSampleJoint );
  this->m_BinaryKNNTreeJoint->GenerateTree();

  /** Initialize tree searchers. */
  this->m_BinaryKNNTreeSearcherFixed
	->SetBinaryTree( this->m_BinaryKNNTreeFixed );
  this->m_BinaryKNNTreeSearcherMoving
	->SetBinaryTree( this->m_BinaryKNNTreeMoving );
  this->m_BinaryKNNTreeSearcherJoint
	->SetBinaryTree( this->m_BinaryKNNTreeJoint );

  /**
   * *************** Estimate the \alpha MI and its derivatives ******************
   *
   * This is done by searching for the nearest neighbours of each point
   * and calculating the distances.
   *
   * The estimate for the alpha - mutual information is given by:
   *
   *  \alpha MI = 1 / ( \alpha - 1 ) * \log 1/n^\alpha * \sum_{i=1}^n \sum_{p=1}^k
   *              ( jointLength / \sqrt( fixedLength * movingLength ) )^(2 \gamma),
   *
   * where
   *   - \alpha is set by the user and refers to \alpha - mutual information
   *   - n is the number of samples
   *   - k is the number of nearest neighbours
   *   - jointLength  is the distances to one of the nearest neighbours in listSampleJoint
   *   - fixedLength  is the distances to one of the nearest neighbours in listSampleFixed
   *   - movingLength is the distances to one of the nearest neighbours in listSampleMoving
   *   - \gamma relates to the distance metric and relates to \alpha as:
   *
   *        \gamma = d * ( 1 - \alpha ),
   *
   *     where d is the dimension of the feature space.
   *
   * In the original paper it is assumed that the mutual information of
   * two feature sets of equal dimension is calculated. If not this is not
   * true, then
   *
   *        \gamma = ( ( d1 + d2 ) / 2 ) * ( 1 - alpha ),
   *
   * where d1 and d2 are the possibly different dimensions of the two feature sets.
   */

  /** Temporary variables. */
  typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
  MeasurementVectorType z_F, z_M, z_J, z_M_ip, z_J_ip, diff_M, diff_J;
  IndexArrayType        indices_F,   indices_M,   indices_J;
  DistanceArrayType     distances_F, distances_M, distances_J;
  MeasureType           distance_F,  distance_M,  distance_J;

  MeasureType    H, G, Gpow;
  AccumulateType sumG = NumericTraits< AccumulateType >::Zero;

  DerivativeType contribution( this->GetNumberOfParameters() );
  contribution.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
  DerivativeType dGamma_M( this->GetNumberOfParameters() );
  DerivativeType dGamma_J( this->GetNumberOfParameters() );

  /** Get the size of the feature vectors. */
  unsigned int fixedSize  = this->GetNumberOfFixedImages();
  unsigned int movingSize = this->GetNumberOfMovingImages();
  unsigned int jointSize  = fixedSize + movingSize;

  /** Get the number of neighbours and \gamma. */
  unsigned int k        = this->m_BinaryKNNTreeSearcherFixed->GetKNearestNeighbors();
  double       twoGamma = jointSize * ( 1.0 - this->m_Alpha );

  /** Loop over all query points, i.e. all samples. */
  for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
  {
	/** Get the i-th query point. */
	listSampleFixed->GetMeasurementVector(  i, z_F );
	listSampleMoving->GetMeasurementVector( i, z_M );
	listSampleJoint->GetMeasurementVector(  i, z_J );

	/** Search for the k nearest neighbours of the current query point. */
	this->m_BinaryKNNTreeSearcherFixed->Search(  z_F, indices_F, distances_F );
	this->m_BinaryKNNTreeSearcherMoving->Search( z_M, indices_M, distances_M );
	this->m_BinaryKNNTreeSearcherJoint->Search(  z_J, indices_J, distances_J );

	/** Variables to compute the measure and its derivative. */
	AccumulateType Gamma_F = NumericTraits< AccumulateType >::Zero;
	AccumulateType Gamma_M = NumericTraits< AccumulateType >::Zero;
	AccumulateType Gamma_J = NumericTraits< AccumulateType >::Zero;

	SpatialDerivativeType D1sparse, D2sparse_M, D2sparse_J;
	D1sparse = spatialDerivativesContainer[ i ] * jacobianContainer[ i ];

	dGamma_M.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
	dGamma_J.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

	/** Loop over the neighbours. */
	for( unsigned int p = 0; p < k; p++ )
	{
	  /** Get the neighbour point z_ip^M. */
	  listSampleMoving->GetMeasurementVector( indices_M[ p ], z_M_ip );
	  listSampleMoving->GetMeasurementVector( indices_J[ p ], z_J_ip );

	  /** Get the distances. */
	  distance_F = vcl_sqrt( distances_F[ p ] );
	  distance_M = vcl_sqrt( distances_M[ p ] );
	  distance_J = vcl_sqrt( distances_J[ p ] );

	  /** Compute Gamma's. */
	  Gamma_F += distance_F;
	  Gamma_M += distance_M;
	  Gamma_J += distance_J;

	  /** Get the difference of z_ip^M with z_i^M. */
	  diff_M = z_M - z_M_ip;
	  diff_J = z_M - z_J_ip;

	  /** Compute derivatives. */
	  D2sparse_M = spatialDerivativesContainer[ indices_M[ p ] ]
		* jacobianContainer[ indices_M[ p ] ];
	  D2sparse_J = spatialDerivativesContainer[ indices_J[ p ] ]
		* jacobianContainer[ indices_J[ p ] ];

	  /** Update the dGamma's. */
	  this->UpdateDerivativeOfGammas(
		D1sparse, D2sparse_M, D2sparse_J,
		jacobianIndicesContainer[ i ],
		jacobianIndicesContainer[ indices_M[ p ] ],
		jacobianIndicesContainer[ indices_J[ p ] ],
		diff_M, diff_J,
		distance_M, distance_J,
		dGamma_M, dGamma_J );

	} // end loop over the k neighbours

	/** Compute contributions. */
	H = vcl_sqrt( Gamma_F * Gamma_M );
	if( H > this->m_AvoidDivisionBy )
	{
	  /** Compute some sums. */
	  G     = Gamma_J / H;
	  sumG += vcl_pow( G, twoGamma );

	  /** Compute the contribution to the derivative. */
	  Gpow          = vcl_pow( G, twoGamma - 1.0 );
	  contribution += ( Gpow / H ) * ( dGamma_J - ( 0.5 * Gamma_J / Gamma_M ) * dGamma_M );
	}

  } // end looping over all query points

  /**
   * *************** Finally, calculate the metric value and derivative ******************
   */

  /** Compute the value. */
  double n, number;
  if( sumG > this->m_AvoidDivisionBy )
  {
	/** Compute the measure. */
	n       = static_cast< double >( this->m_NumberOfPixelsCounted );
	number  = vcl_pow( n, this->m_Alpha );
	measure = vcl_log( sumG / number ) / ( this->m_Alpha - 1.0 );

	/** Compute the derivative (-2.0 * d = -jointSize). */
	derivative = ( static_cast< AccumulateType >( jointSize ) / sumG ) * contribution;
  }
  value = -measure;

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ************************ GetValueAndDerivative *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** Option for now to still use the single threaded code. */
  if( !this->m_UseKNNMultiThread )
  {
	return this->GetValueAndDerivativeSingleThreaded(
			     parameters, value, derivative );
  }
	
  /** Initialize some variables. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Initialize some threading related parameters. */
  this->InitializeKNNThreadingParameters();

  /**
   * *************** Evoking the multi-threads for the list samples ******************
   *
   * and the derivatives plus jacobian.
   */
  this->m_KNNThreader->SetNumberOfThreads( this->m_KNNNumberOfThreads );
  this->m_KNNThreader->SetSingleMethod( KNNListSamplesAndDerivativePlusJacobianThreaderCallback, 
	const_cast<void *>(static_cast<const void *>(&this->m_KNNThreaderMetricParameters)) );
  this->m_KNNThreader->SingleMethodExecute();

  /** Collect listsamples and derivatives plus jacobian from all threads. */
  this->AfterKNNThreadedListSamplesAndDerivativePlusJacobian();
  
  /**
   * *************** Generate the three trees ******************
   *
   * and connect them to the searchers.
   */

  /** Generate the tree for the fixed image samples. */
  this->m_BinaryKNNTreeFixed->SetSample(this->m_KNNThreaderMetricParameters.st_FixedListSample);
  this->m_BinaryKNNTreeFixed->GenerateTree();

  /** Generate the tree for the moving image samples. */
  this->m_BinaryKNNTreeMoving->SetSample(this->m_KNNThreaderMetricParameters.st_MovingListSample);
  this->m_BinaryKNNTreeMoving->GenerateTree();

  /** Generate the tree for the joint image samples. */
  this->m_BinaryKNNTreeJoint->SetSample(this->m_KNNThreaderMetricParameters.st_JointListSample);
  this->m_BinaryKNNTreeJoint->GenerateTree();

  /** Initialize tree searchers. */
  this->m_BinaryKNNTreeSearcherFixed
  ->SetBinaryTree( this->m_BinaryKNNTreeFixed );
  this->m_BinaryKNNTreeSearcherMoving
  ->SetBinaryTree( this->m_BinaryKNNTreeMoving );
  this->m_BinaryKNNTreeSearcherJoint
  ->SetBinaryTree( this->m_BinaryKNNTreeJoint );

  /**
   * *************** Estimate the \alpha MI and its derivatives ******************
   *
   * This is done by searching for the nearest neighbours of each point
   * and calculating the distances.
   *
   * The estimate for the alpha - mutual information is given by:
   *
   *  \alpha MI = 1 / ( \alpha - 1 ) * \log 1/n^\alpha * \sum_{i=1}^n \sum_{p=1}^k
   *              ( jointLength / \sqrt( fixedLength * movingLength ) )^(2 \gamma),
   *
   * where
   *   - \alpha is set by the user and refers to \alpha - mutual information
   *   - n is the number of samples
   *   - k is the number of nearest neighbours
   *   - jointLength  is the distances to one of the nearest neighbours in listSampleJoint
   *   - fixedLength  is the distances to one of the nearest neighbours in listSampleFixed
   *   - movingLength is the distances to one of the nearest neighbours in listSampleMoving
   *   - \gamma relates to the distance metric and relates to \alpha as:
   *
   *        \gamma = d * ( 1 - \alpha ),
   *
   *     where d is the dimension of the feature space.
   *
   * In the original paper it is assumed that the mutual information of
   * two feature sets of equal dimension is calculated. If not this is not
   * true, then
   *
   *        \gamma = ( ( d1 + d2 ) / 2 ) * ( 1 - alpha ),
   *
   * where d1 and d2 are the possibly different dimensions of the two feature sets.
   */

  /** Temporary variables. */
  typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
  MeasurementVectorType z_F, z_M, z_J;
  IndexArrayType        indices_F,   indices_M,   indices_J;
  DistanceArrayType     distances_F, distances_M, distances_J;

  AccumulateType sumG = NumericTraits< AccumulateType >::Zero;

  this->m_KNNThreaderMetricParameters.st_MovingIndexArrayContainer.reserve( this->m_NumberOfPixelsCounted );
  this->m_KNNThreaderMetricParameters.st_JointIndexArrayContainer.reserve( this->m_NumberOfPixelsCounted );
  this->m_KNNThreaderMetricParameters.st_FixedDistArrayContainer.reserve( this->m_NumberOfPixelsCounted );
  this->m_KNNThreaderMetricParameters.st_MovingDistArrayContainer.reserve( this->m_NumberOfPixelsCounted );
  this->m_KNNThreaderMetricParameters.st_JointDistArrayContainer.reserve( this->m_NumberOfPixelsCounted );

  /** Get the size of the feature vectors. */
  unsigned int fixedSize  = this->GetNumberOfFixedImages();
  unsigned int movingSize = this->GetNumberOfMovingImages();
  unsigned int jointSize  = fixedSize + movingSize;

  this->m_KNNThreaderMetricParameters.st_KNearestNeighbours = this->m_BinaryKNNTreeSearcherFixed->GetKNearestNeighbors();
  /** Loop over all query points, i.e. all samples. */
  for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; ++i )
  {
	/** Get the i-th query point. */
	this->m_KNNThreaderMetricParameters.st_FixedListSample->GetMeasurementVector(i, z_F);
	this->m_KNNThreaderMetricParameters.st_MovingListSample->GetMeasurementVector(i, z_M);
	this->m_KNNThreaderMetricParameters.st_JointListSample->GetMeasurementVector(i, z_J);

	/** Search for the k nearest neighbours of the current query point. */
	this->m_BinaryKNNTreeSearcherFixed->Search(  z_F, indices_F, distances_F );
	this->m_BinaryKNNTreeSearcherMoving->Search( z_M, indices_M, distances_M );
	this->m_BinaryKNNTreeSearcherJoint->Search(  z_J, indices_J, distances_J );

	/** Transfer information of the k nearest neighbours for multi-threading. */
	this->m_KNNThreaderMetricParameters.st_MovingIndexArrayContainer.push_back( indices_M );
	this->m_KNNThreaderMetricParameters.st_JointIndexArrayContainer.push_back( indices_J );
	this->m_KNNThreaderMetricParameters.st_FixedDistArrayContainer.push_back( distances_F );
	this->m_KNNThreaderMetricParameters.st_MovingDistArrayContainer.push_back( distances_M );
	this->m_KNNThreaderMetricParameters.st_JointDistArrayContainer.push_back( distances_J );
	
  } // end looping over all query points

  /**
   * *************** Evoking the multi-threads for the value and derivative ******************
   */
  this->m_KNNThreader->SetNumberOfThreads( this->m_KNNNumberOfThreads );
  this->m_KNNThreader->SetSingleMethod( KNNGetValueAndDerivativeThreaderCallback, 
	const_cast<void *>(static_cast<const void *>(&this->m_KNNThreaderMetricParameters)) );
  this->m_KNNThreader->SingleMethodExecute();

  /** Collect the derivative using multi-threading. */
  /*for( ThreadIdType i = 0; i < this->m_KNNNumberOfThreads; ++i )
  {
	this->m_KNNThreaderMetricParameters.st_KNNContrib += this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_contribution;
  }*/
  this->m_KNNThreader->SetNumberOfThreads( this->m_KNNNumberOfThreads );
  this->m_KNNThreader->SetSingleMethod( KNNMergeDerivativeThreaderCallback, 
	const_cast<void *>(static_cast<const void *>(&this->m_KNNThreaderMetricParameters)) );
  this->m_KNNThreader->SingleMethodExecute();

  /**
   * *************** Finally, calculate the metric value and derivative ******************
   */ 

  /** Collect the value from all threads. */
  for( ThreadIdType i = 0; i < this->m_KNNNumberOfThreads; ++i )
  {
	sumG += this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_sumG;

	this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_jacobianContainer.swap( TransformJacobianContainerType() );
	this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_jacobianIndicesContainer.swap( TransformJacobianIndicesContainerType() );
	this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_spatialDerivativesContainer.swap( SpatialDerivativeContainerType() );	
  }

  /** Compute the value. */
  double n, number;
  if( sumG > this->m_AvoidDivisionBy )
  {
	/** Compute the measure. */
	n       = static_cast< double >( this->m_NumberOfPixelsCounted );
	number  = vcl_pow( n, this->m_Alpha );
	measure = vcl_log( sumG / number ) / ( this->m_Alpha - 1.0 );

	/** Compute the derivative (-2.0 * d = -jointSize). */
	derivative = ( static_cast< AccumulateType >( jointSize ) / sumG ) * this->m_KNNThreaderMetricParameters.st_KNNContrib;
  }

  value = -measure;

  /** Release some space for next iteration. */
  this->m_KNNThreaderMetricParameters.st_JacobianContainer.swap(TransformJacobianContainerType());
  this->m_KNNThreaderMetricParameters.st_JacobianIndicesContainer.swap(TransformJacobianIndicesContainerType());
  this->m_KNNThreaderMetricParameters.st_SpatialDerivativesContainer.swap(SpatialDerivativeContainerType());
  this->m_KNNThreaderMetricParameters.st_MovingIndexArrayContainer.swap( IndexArrayContainerType() );
  this->m_KNNThreaderMetricParameters.st_JointIndexArrayContainer.swap( IndexArrayContainerType() );
  this->m_KNNThreaderMetricParameters.st_FixedDistArrayContainer.swap( DistanceArrayContainerType() ); 
  this->m_KNNThreaderMetricParameters.st_MovingDistArrayContainer.swap( DistanceArrayContainerType() ); 
  this->m_KNNThreaderMetricParameters.st_JointDistArrayContainer.swap( DistanceArrayContainerType() );

} // end GetValueAndDerivative()


/**
 * ************************ ComputeListSampleValuesAndDerivativePlusJacobian *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::ComputeListSampleValuesAndDerivativePlusJacobian(
  const ListSamplePointer & listSampleFixed,
  const ListSamplePointer & listSampleMoving,
  const ListSamplePointer & listSampleJoint,
  const bool & doDerivative,
  TransformJacobianContainerType & jacobianContainer,
  TransformJacobianIndicesContainerType & jacobianIndicesContainer,
  SpatialDerivativeContainerType & spatialDerivativesContainer ) const
{
  /** Initialize. */
  this->m_NumberOfPixelsCounted = 0;
  jacobianContainer.resize( 0 );
  jacobianIndicesContainer.resize( 0 );
  spatialDerivativesContainer.resize( 0 );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer      = this->GetImageSampler()->GetOutput();
  const unsigned long         nrOfRequestedSamples = sampleContainer->Size();

  /** Create an iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /** Resize the list samples so that enough memory is allocated. */
  listSampleFixed->SetMeasurementVectorSize( fixedSize );
  listSampleFixed->Resize( nrOfRequestedSamples );
  listSampleMoving->SetMeasurementVectorSize( movingSize );
  listSampleMoving->Resize( nrOfRequestedSamples );
  listSampleJoint->SetMeasurementVectorSize( jointSize );
  listSampleJoint->Resize( nrOfRequestedSamples );

  /** Potential speedup: it avoids re-allocations. I noticed performance
   * gains when nrOfRequestedSamples is about 10000 or higher.
   */
  jacobianContainer.reserve( nrOfRequestedSamples );
  jacobianIndicesContainer.reserve( nrOfRequestedSamples );
  spatialDerivativesContainer.reserve( nrOfRequestedSamples );

  /** Create variables to store intermediate results. */
  RealType                   movingImageValue;
  MovingImagePointType       mappedPoint;
  double                     fixedFeatureValue  = 0.0;
  double                     movingFeatureValue = 0.0;
  NonZeroJacobianIndicesType nzji(
  this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  TransformJacobianType jacobian;

  /** Loop over the fixed image samples to calculate the list samples. */
  unsigned int ii = 0;
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside all moving masks. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value M(T(x)) and possibly the
     * derivative dM/dx and check if the point is inside all
     * moving images buffers.
     */
    MovingImageDerivativeType movingImageDerivative;
    if( sampleOk )
    {
      if( doDerivative )
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, &movingImageDerivative );
      }
      else
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, 0 );
      }
    }

    /** This is a valid sample: in this if-statement the actual
     * addition to the list samples is done.
     */
    if( sampleOk )
    {
      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast< RealType >(
        ( *fiter ).Value().m_ImageValue );

      /** Add the samples to the ListSampleCarrays. */
      listSampleFixed->SetMeasurement(  this->m_NumberOfPixelsCounted, 0,
        fixedImageValue );
      listSampleMoving->SetMeasurement( this->m_NumberOfPixelsCounted, 0,
        movingImageValue );
      listSampleJoint->SetMeasurement(  this->m_NumberOfPixelsCounted, 0,
        fixedImageValue );
      listSampleJoint->SetMeasurement(  this->m_NumberOfPixelsCounted,
        this->GetNumberOfFixedImages(), movingImageValue );

      /** Get and set the values of the fixed feature images. */
      for( unsigned int j = 1; j < this->GetNumberOfFixedImages(); j++ )
      {
        fixedFeatureValue = this->m_FixedImageInterpolatorVector[ j ]
          ->Evaluate( fixedPoint );
        listSampleFixed->SetMeasurement(
          this->m_NumberOfPixelsCounted, j, fixedFeatureValue );
        listSampleJoint->SetMeasurement(
          this->m_NumberOfPixelsCounted, j, fixedFeatureValue );
      }

      /** Get and set the values of the moving feature images. */
      for( unsigned int j = 1; j < this->GetNumberOfMovingImages(); j++ )
      {
        movingFeatureValue = this->m_InterpolatorVector[ j ]
          ->Evaluate( mappedPoint );
        listSampleMoving->SetMeasurement(
          this->m_NumberOfPixelsCounted,
          j,
          movingFeatureValue );
        listSampleJoint->SetMeasurement(
          this->m_NumberOfPixelsCounted,
          j + this->GetNumberOfFixedImages(),
          movingFeatureValue );
      }

      /** Compute additional stuff for the computation of the derivative, if necessary.
       * - the Jacobian of the transform: dT/dmu(x_i).
       * - the spatial derivative of all moving feature images: dz_q^m/dx(T(x_i)).
       */
      if( doDerivative )
      {
        /** Get the TransformJacobian dT/dmu. */
        this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
        jacobianContainer.push_back( jacobian );
        jacobianIndicesContainer.push_back( nzji );

        /** Get the spatial derivative of the moving image. */
        SpatialDerivativeType spatialDerivatives(
        this->GetNumberOfMovingImages(),
        this->FixedImageDimension );
        spatialDerivatives.set_row( 0, movingImageDerivative.GetDataPointer() );

        /** Get the spatial derivatives of the moving feature images. */
        SpatialDerivativeType movingFeatureImageDerivatives(
        this->GetNumberOfMovingImages() - 1,
        this->FixedImageDimension );
        this->EvaluateMovingFeatureImageDerivatives(
          mappedPoint, movingFeatureImageDerivatives );
        spatialDerivatives.update( movingFeatureImageDerivatives, 1, 0 );

        /** Put the spatial derivatives of this sample into the container. */
        spatialDerivativesContainer.push_back( spatialDerivatives );

      } // end if doDerivative

      /** Update the NumberOfPixelsCounted. */
      this->m_NumberOfPixelsCounted++;

      ii++;

    } // end if sampleOk

  } // end for loop over the image sample container

  /** The listSamples are of size sampleContainer->Size(). However, not all of
   * those points made it to the respective list samples. Therefore, we set
   * the actual number of pixels in the sample container, so that the binary
   * trees know where to loop over. This must not be forgotten!
   */
  listSampleFixed->SetActualSize( this->m_NumberOfPixelsCounted );
  listSampleMoving->SetActualSize( this->m_NumberOfPixelsCounted );
  listSampleJoint->SetActualSize( this->m_NumberOfPixelsCounted );

} // end ComputeListSampleValuesAndDerivativePlusJacobian()


/**
 * ************************ EvaluateMovingFeatureImageDerivatives *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::EvaluateMovingFeatureImageDerivatives(
  const MovingImagePointType & mappedPoint,
  SpatialDerivativeType & featureGradients ) const
{
  /** Convert point to a continous index. */
  MovingImageContinuousIndexType cindex;
  this->m_Interpolator->ConvertPointToContinuousIndex( mappedPoint, cindex );

  /** Compute the spatial derivative for all feature images:
   * - either by calling a special function that only B-spline
   *   interpolators have,
   * - or by using a finite difference approximation of the
   *   pre-computed gradient images.
   * \todo: for now we only implement the first option.
   */
  if( this->m_InterpolatorsAreBSpline && !this->GetComputeGradient() )
  {
    /** Computed moving image gradient using derivative B-spline kernel. */
    MovingImageDerivativeType gradient;
    for( unsigned int i = 1; i < this->GetNumberOfMovingImages(); ++i )
    {
      /** Compute the gradient at feature image i. */
      gradient = this->m_BSplineInterpolatorVector[ i ]
        ->EvaluateDerivativeAtContinuousIndex( cindex );

      /** Set the gradient into the Array2D. */
      featureGradients.set_row( i - 1, gradient.GetDataPointer() );
    } // end for-loop
  }   // end if
//  else
//  {
//  /** Get the gradient by NearestNeighboorInterpolation of the gradient image.
//  * It is assumed that the gradient image is computed beforehand.
//  */
//
//  /** Round the continuous index to the nearest neighbour. */
//  MovingImageIndexType index;
//  for ( unsigned int j = 0; j < MovingImageDimension; j++ )
//  {
//  index[ j ] = static_cast<long>( vnl_math_rnd( cindex[ j ] ) );
//  }
//
//  MovingImageDerivativeType gradient;
//  for ( unsigned int i = 0; i < this->m_NumberOfMovingFeatureImages; ++i )
//  {
//  /** Compute the gradient at feature image i. */
//  gradient = this->m_GradientFeatureImage[ i ]->GetPixel( index );
//
//  /** Set the gradient into the Array2D. */
//  featureGradients.set_column( i, gradient.GetDataPointer() );
//  } // end for-loop
//  } // end if

} // end EvaluateMovingFeatureImageDerivatives()


/**
 * **************** KNNListSamplesAndDerivativePlusJacobianThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::KNNListSamplesAndDerivativePlusJacobianThreaderCallback( void * arg )
{
  KNNThreadInfoType * infoStruct = static_cast< KNNThreadInfoType * >( arg );
  ThreadIdType        threadID   = infoStruct->ThreadID;

  KNNMultiThreaderParameterType * temp
	= static_cast< KNNMultiThreaderParameterType * >( infoStruct->UserData );

  temp->st_Metric->KNNThreadedListSamplesAndDerivativePlusJacobian( threadID );

  return ITK_THREAD_RETURN_VALUE;

} // end KNNListSamplesAndDerivativePlusJacobianThreaderCallback()


/**
 * ******************* KNNThreadedListSamplesAndDerivativePlusJacobian *******************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::KNNThreadedListSamplesAndDerivativePlusJacobian( ThreadIdType threadId )
{
  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long sampleContainerSize     = sampleContainer->Size();

  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
	= static_cast< unsigned long >( vcl_ceil( static_cast< double >( sampleContainerSize )
	/ static_cast< double >( this->m_KNNNumberOfThreads ) ) );

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
  pos_end   = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

  /** Resize the list samples so that enough memory is allocated. */
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->SetMeasurementVectorSize( fixedSize );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->Resize( nrOfSamplesPerThreads );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->SetMeasurementVectorSize( movingSize );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->Resize( nrOfSamplesPerThreads );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurementVectorSize( jointSize );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->Resize( nrOfSamplesPerThreads );

  /** Potential speedup: it avoids re-allocations. I noticed performance
   * gains when nrOfRequestedSamples is about 10000 or higher.
   */
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jacobianContainer.reserve( nrOfSamplesPerThreads );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jacobianIndicesContainer.reserve( nrOfSamplesPerThreads );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_spatialDerivativesContainer.reserve( nrOfSamplesPerThreads ); 

  /** Create variables to store intermediate results. */
  RealType movingImageValue;
  MovingImagePointType mappedPoint;
  double fixedFeatureValue = 0.0;
  double movingFeatureValue = 0.0;
  NonZeroJacobianIndicesType nzji(
	this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  TransformJacobianType jacobian;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend   = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend   += (int)pos_end;

  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted = 0;
  /** Loop over the fixed image samples to calculate the list samples. */
  for( threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter )
  {
	/** Read fixed coordinates and initialize some variables. */
	const FixedImagePointType & fixedPoint = (*threader_fiter).Value().m_ImageCoordinates;

	/** Transform point and check if it is inside the B-spline support region. */
	bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

	/** Check if point is inside all moving masks. */
	if( sampleOk )
	{
	  sampleOk = this->IsInsideMovingMask( mappedPoint );
	}

	/** Compute the moving image value M(T(x)) and possibly the
	 * derivative dM/dx and check if the point is inside all
	 * moving images buffers.
	 */
	MovingImageDerivativeType movingImageDerivative;
	if( sampleOk )
	{
	  sampleOk = this->EvaluateMovingImageValueAndDerivative(
		mappedPoint, movingImageValue, &movingImageDerivative );
	}

	/** This is a valid sample: in this if-statement the actual
	 * addition to the list samples is done.
	 */
	if( sampleOk )
	{
	  /** Get the fixed image value. */
	  const RealType & fixedImageValue = static_cast<RealType>(
		(*threader_fiter).Value().m_ImageValue );

	  /** Add the samples to the ListSamplearrays. */
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->SetMeasurement( 
		this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, 0, fixedImageValue );
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->SetMeasurement( 
		this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, 0, movingImageValue );
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurement( 
		this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, 0, fixedImageValue );
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurement( 
		this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, fixedSize, movingImageValue );

	  /** Get and set the values of the fixed feature images. */
	  for( unsigned int j = 1; j < fixedSize; ++j )
	  {
		fixedFeatureValue = this->m_FixedImageInterpolatorVector[ j ]->Evaluate( fixedPoint );
		this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->SetMeasurement( 
		  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j, fixedFeatureValue );
		this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurement( 
		  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j, fixedFeatureValue );
	  }

	  /** Get and set the values of the moving feature images. */
	  for( unsigned int j = 1; j < movingSize; ++j )
	  {
		movingFeatureValue = this->m_InterpolatorVector[ j ]->Evaluate( mappedPoint );
		this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->SetMeasurement( 
		  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j, movingFeatureValue );
		this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetMeasurement( 
		  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted, j + fixedSize, movingFeatureValue );
	  }

	  /** Compute additional stuff for the computation of the derivative, if necessary.
	   * - the Jacobian of the transform: dT/dmu(x_i).
	   * - the spatial derivative of all moving feature images: dz_q^m/dx(T(x_i)).
	   */

	  /** Get the TransformJacobian dT/dmu. */
	  this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jacobianContainer.push_back( jacobian );
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jacobianIndicesContainer.push_back( nzji );

	  /** Get the spatial derivative of the moving image. */
	  SpatialDerivativeType spatialDerivatives(
		this->GetNumberOfMovingImages(),
		this->FixedImageDimension );
	  spatialDerivatives.set_row( 0, movingImageDerivative.GetDataPointer() );

	  /** Get the spatial derivatives of the moving feature images. */
	  SpatialDerivativeType movingFeatureImageDerivatives(
		this->GetNumberOfMovingImages() - 1,
		this->FixedImageDimension );
	  this->EvaluateMovingFeatureImageDerivatives(
		mappedPoint, movingFeatureImageDerivatives );
	  spatialDerivatives.update( movingFeatureImageDerivatives, 1, 0 );

	  /** Put the spatial derivatives of this sample into the container. */
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_spatialDerivativesContainer.push_back( spatialDerivatives ); 

	  /** Update the NumberOfPixelsCounted for this thread. */
	  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted++;

	} // end if sampleOk

  } // end for loop over the image sample container

  /** The listSamples are of size sampleContainer->Size(). However, not all of
   * those points made it to the respective list samples. Therefore, we set
   * the actual number of pixels in the sample container, so that the binary
   * trees know where to loop over. This must not be forgotten!
   */
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_fixedListSample->SetActualSize( 
	this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_movingListSample->SetActualSize( 
	this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted );
  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_jointListSample->SetActualSize( 
	this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_numberOfPixelsCounted );

} // end KNNThreadedListSamplesAndDerivativePlusJacobian()


/**
 * ******************* AfterKNNThreadedListSamplesAndDerivativePlusJacobian *******************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::AfterKNNThreadedListSamplesAndDerivativePlusJacobian( void ) const
{
  /** Gather the NumberOfPixelsCounted from all threads. */
  this->m_NumberOfPixelsCounted = 0;
  for( ThreadIdType i = 0; i < this->m_KNNNumberOfThreads; ++i )
	this->m_NumberOfPixelsCounted += this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_numberOfPixelsCounted;

  /** Check if enough samples were valid. */
  unsigned long numSamplePoints = this->GetImageSampler()->GetOutput()->Size();
  this->CheckNumberOfSamples( numSamplePoints, this->m_NumberOfPixelsCounted );

  /** Get the size of the feature vectors. */
  const unsigned int fixedSize  = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize  = fixedSize + movingSize;

  /** Resize the list samples so that enough memory is allocated. */
  this->m_KNNThreaderMetricParameters.st_FixedListSample->SetMeasurementVectorSize(fixedSize);
  this->m_KNNThreaderMetricParameters.st_FixedListSample->Resize(this->m_NumberOfPixelsCounted);
  this->m_KNNThreaderMetricParameters.st_FixedListSample->SetActualSize(this->m_NumberOfPixelsCounted);
  this->m_KNNThreaderMetricParameters.st_MovingListSample->SetMeasurementVectorSize(movingSize);
  this->m_KNNThreaderMetricParameters.st_MovingListSample->Resize(this->m_NumberOfPixelsCounted);
  this->m_KNNThreaderMetricParameters.st_MovingListSample->SetActualSize(this->m_NumberOfPixelsCounted);
  this->m_KNNThreaderMetricParameters.st_JointListSample->SetMeasurementVectorSize(jointSize);
  this->m_KNNThreaderMetricParameters.st_JointListSample->Resize(this->m_NumberOfPixelsCounted);
  this->m_KNNThreaderMetricParameters.st_JointListSample->SetActualSize(this->m_NumberOfPixelsCounted);

  /** Potential speedup: it avoids re-allocations. I noticed performance
   * gains when nrOfRequestedSamples is about 10000 or higher.
   */
  this->m_KNNThreaderMetricParameters.st_JacobianContainer.reserve(this->m_NumberOfPixelsCounted);
  this->m_KNNThreaderMetricParameters.st_JacobianIndicesContainer.reserve(this->m_NumberOfPixelsCounted);
  this->m_KNNThreaderMetricParameters.st_SpatialDerivativesContainer.reserve(this->m_NumberOfPixelsCounted);

  /** Gather listsamples and derivatives plus jacobian from all threads. */
  unsigned long nOPC = 0;
  for( ThreadIdType i = 0; i < this->m_KNNNumberOfThreads; ++i )
  {
	unsigned long nOPCThread = this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_numberOfPixelsCounted;
	for( unsigned long j = 0; j < nOPCThread; ++j )
	{
      this->m_KNNThreaderMetricParameters.st_FixedListSample->SetMeasurementVector(
		nOPC, this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_fixedListSample->GetMeasurementVector( j ) );
	  this->m_KNNThreaderMetricParameters.st_MovingListSample->SetMeasurementVector(
		nOPC, this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_movingListSample->GetMeasurementVector( j ) );
	  this->m_KNNThreaderMetricParameters.st_JointListSample->SetMeasurementVector(
		nOPC, this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_jointListSample->GetMeasurementVector( j ) );
				
	  this->m_KNNThreaderMetricParameters.st_JacobianContainer.push_back(this->m_KNNGetValueAndDerivativePerThreadVariables[i].st_jacobianContainer[j]);
	  this->m_KNNThreaderMetricParameters.st_JacobianIndicesContainer.push_back(this->m_KNNGetValueAndDerivativePerThreadVariables[i].st_jacobianIndicesContainer[j]);
	  this->m_KNNThreaderMetricParameters.st_SpatialDerivativesContainer.push_back(this->m_KNNGetValueAndDerivativePerThreadVariables[i].st_spatialDerivativesContainer[j]);
	  nOPC++;
	}
  }

} // end AfterKNNThreadedListSamplesAndDerivativePlusJacobian()


/**
 * **************** KNNGetValueAndDerivativeThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::KNNGetValueAndDerivativeThreaderCallback( void * arg )
{
  KNNThreadInfoType * infoStruct = static_cast< KNNThreadInfoType * >( arg );
  ThreadIdType        threadID   = infoStruct->ThreadID;

  KNNMultiThreaderParameterType * temp
	= static_cast< KNNMultiThreaderParameterType * >( infoStruct->UserData );

  temp->st_Metric->KNNThreadedGetValueAndDerivative( threadID );

  return ITK_THREAD_RETURN_VALUE;

} // end KNNGetValueAndDerivativeThreaderCallback()


/**
 * ******************* KNNThreadedGetValueAndDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::KNNThreadedGetValueAndDerivative( ThreadIdType threadId ) 
{
  /** Temporary variables. */
  typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
  MeasurementVectorType z_M, z_M_ip, z_J_ip, diff_M, diff_J;
  MeasureType           distance_F,  distance_M,  distance_J;

  MeasureType    H, G, Gpow;
  AccumulateType sumG = NumericTraits< AccumulateType >::Zero;

  DerivativeType dGamma_M( this->GetNumberOfParameters() );
  DerivativeType dGamma_J( this->GetNumberOfParameters() );

  /** Get the size of the feature vectors. */
  unsigned int fixedSize  = this->GetNumberOfFixedImages();
  unsigned int movingSize = this->GetNumberOfMovingImages();
  unsigned int jointSize  = fixedSize + movingSize;

  /** Get the number of neighbours and \gamma. */
  unsigned int k        = this->m_KNNThreaderMetricParameters.st_KNearestNeighbours;
  double       twoGamma = jointSize * ( 1.0 - this->m_Alpha );

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
	= static_cast< unsigned long >( vcl_ceil( static_cast< double >( this->m_NumberOfPixelsCounted )
	/ static_cast< double >( this->m_KNNNumberOfThreads ) ) );

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > this->m_NumberOfPixelsCounted ) ? this->m_NumberOfPixelsCounted : pos_begin;
  pos_end   = ( pos_end > this->m_NumberOfPixelsCounted ) ? this->m_NumberOfPixelsCounted : pos_end;

  /** Get a handle to the pre-allocated derivative for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & dKNN = this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_contribution;

  /** Loop over all query points, i.e. all samples. */
  for( unsigned long i = pos_begin; i < pos_end; ++i )
  {
	this->m_KNNThreaderMetricParameters.st_MovingListSample->GetMeasurementVector(i, z_M);

	/** Variables to compute the measure and its derivative. */
	AccumulateType Gamma_F = NumericTraits< AccumulateType >::Zero;
	AccumulateType Gamma_M = NumericTraits< AccumulateType >::Zero;
	AccumulateType Gamma_J = NumericTraits< AccumulateType >::Zero;

	SpatialDerivativeType D1sparse, D2sparse_M, D2sparse_J;
	D1sparse = this->m_KNNThreaderMetricParameters.st_SpatialDerivativesContainer[i] *
	  this->m_KNNThreaderMetricParameters.st_JacobianContainer[i];

	dGamma_M.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
	dGamma_J.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

	/** Loop over the neighbours. */
	for( unsigned int p = 0; p < k; ++p )
	{
	  /** Get the neighbour point z_ip^M. */
	  this->m_KNNThreaderMetricParameters.st_MovingListSample->GetMeasurementVector(
		this->m_KNNThreaderMetricParameters.st_MovingIndexArrayContainer[i][p], z_M_ip );
	  this->m_KNNThreaderMetricParameters.st_MovingListSample->GetMeasurementVector(
		this->m_KNNThreaderMetricParameters.st_JointIndexArrayContainer[i][p], z_J_ip );

	  /** Get the distances. */
	  distance_F = vcl_sqrt( this->m_KNNThreaderMetricParameters.st_FixedDistArrayContainer[i][p] );
	  distance_M = vcl_sqrt( this->m_KNNThreaderMetricParameters.st_MovingDistArrayContainer[i][p] );
	  distance_J = vcl_sqrt( this->m_KNNThreaderMetricParameters.st_JointDistArrayContainer[i][p] );

	  /** Compute Gamma's. */
	  Gamma_F += distance_F;
	  Gamma_M += distance_M;
	  Gamma_J += distance_J;

	  /** Get the difference of z_ip^M with z_i^M. */
	  diff_M = z_M - z_M_ip;
	  diff_J = z_M - z_J_ip;

	  /** Compute derivatives. */
	  D2sparse_M = this->m_KNNThreaderMetricParameters.st_SpatialDerivativesContainer[this->m_KNNThreaderMetricParameters.st_MovingIndexArrayContainer[i][p]]
		* this->m_KNNThreaderMetricParameters.st_JacobianContainer[this->m_KNNThreaderMetricParameters.st_MovingIndexArrayContainer[i][p]];
	  D2sparse_J = this->m_KNNThreaderMetricParameters.st_SpatialDerivativesContainer[this->m_KNNThreaderMetricParameters.st_JointIndexArrayContainer[i][p]]
		* this->m_KNNThreaderMetricParameters.st_JacobianContainer[this->m_KNNThreaderMetricParameters.st_JointIndexArrayContainer[i][p]];

	  /** Update the dGamma's. */
	  this->UpdateDerivativeOfGammas(
		D1sparse, D2sparse_M, D2sparse_J,
		this->m_KNNThreaderMetricParameters.st_JacobianIndicesContainer[i],
		this->m_KNNThreaderMetricParameters.st_JacobianIndicesContainer[this->m_KNNThreaderMetricParameters.st_MovingIndexArrayContainer[i][p]],
		this->m_KNNThreaderMetricParameters.st_JacobianIndicesContainer[this->m_KNNThreaderMetricParameters.st_JointIndexArrayContainer[i][p]],
		diff_M, diff_J,
		distance_M, distance_J,
		dGamma_M, dGamma_J );

	} // end loop over the k neighbours

	/** Compute contributions. */
	H = vcl_sqrt( Gamma_F * Gamma_M );
	if( H > this->m_AvoidDivisionBy )
	{
	  /** Compute some sums. */
	  G     = Gamma_J / H;
	  sumG += vcl_pow( G, twoGamma );

	  /** Compute the contribution to the derivative. */
	  Gpow          = vcl_pow( G, twoGamma - 1.0 );
	  dKNN += ( Gpow / H ) * ( dGamma_J - ( 0.5 * Gamma_J / Gamma_M ) * dGamma_M );
	}

  } // end looping over all query points

  this->m_KNNGetValueAndDerivativePerThreadVariables[ threadId ].st_sumG = sumG;

} // end KNNThreadedGetValueAndDerivative()


/**
 * **************** KNNMergeDerivativeThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::KNNMergeDerivativeThreaderCallback( void * arg )
{
  KNNThreadInfoType * infoStruct  = static_cast< KNNThreadInfoType * >( arg );
  ThreadIdType        threadID    = infoStruct->ThreadID;

  KNNMultiThreaderParameterType * temp
	= static_cast< KNNMultiThreaderParameterType * >( infoStruct->UserData );

  temp->st_Metric->KNNThreadedMergeDerivative( threadID );

  return ITK_THREAD_RETURN_VALUE;

} // end KNNMergeDerivativeThreaderCallback()


/**
* ******************* KNNThreadedMergeDerivative *******************
*/

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::KNNThreadedMergeDerivative( ThreadIdType threadId ) 
{
  const unsigned long numPar  = this->GetNumberOfParameters();
  const unsigned long subSize = static_cast< unsigned int >(
		vcl_ceil( static_cast< double >( numPar )
		/ static_cast< double >( this->m_KNNNumberOfThreads ) ) );
  unsigned long jmin = threadId * subSize;
  unsigned long jmax = ( threadId + 1 ) * subSize;
  jmin = ( jmin > numPar ) ? numPar : jmin;
  jmax = ( jmax > numPar ) ? numPar : jmax;

  /** This thread accumulates all sub-derivatives into a single one, for the
   * range [ jmin, jmax [. Additionally, the sub-derivatives are reset.
   */
  for( unsigned long j = jmin; j < jmax; ++j )
  {
	for( ThreadIdType i = 0; i < this->m_KNNNumberOfThreads; ++i )
	{
	  this->m_KNNThreaderMetricParameters.st_KNNContrib[ j ] += this->m_KNNGetValueAndDerivativePerThreadVariables[ i ].st_contribution[ j ];
	}
  }

} // end KNNThreadedMergeDerivative()


/**
 * ************************ UpdateDerivativeOfGammas *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::UpdateDerivativeOfGammas(
  const SpatialDerivativeType & D1sparse,
  const SpatialDerivativeType & D2sparse_M,
  const SpatialDerivativeType & D2sparse_J,
  const NonZeroJacobianIndicesType & D1indices,
  const NonZeroJacobianIndicesType & D2indices_M,
  const NonZeroJacobianIndicesType & D2indices_J,
  const MeasurementVectorType & diff_M,
  const MeasurementVectorType & diff_J,
  const MeasureType & distance_M,
  const MeasureType & distance_J,
  DerivativeType & dGamma_M,
  DerivativeType & dGamma_J ) const
{
  /** Make temporary copies of diff, since post_multiply changes diff. */
  vnl_vector< double > tmpM1( diff_M );
  vnl_vector< double > tmpM2( diff_M );
  vnl_vector< double > tmpJ1( diff_J );
  vnl_vector< double > tmpJ2( diff_J );

  /** Divide by the distance first, so that diff's are normalised.
   * Dividing at this place is much faster, since distance_? is a small
   * vector, i.e. only the size of the number of features (e.g. 6).
   * Dividing tmp?sparse_? is slower, since it is a vector of the size of
   * the B-spline support, so in 3D and spline order 3: (3 + 1)^3 * 3 = 192.
   * On an example registration it gave me a speedup of about 25%!
   * Both methods should return the same results, but due to numerical
   * stuff the metric value and derivative start to deviate after a couple
   * of iterations.
   */
  if( distance_M > this->m_AvoidDivisionBy )
  {
    tmpM1 /= distance_M;
    tmpM2 /= distance_M;
  }
  if( distance_J > this->m_AvoidDivisionBy )
  {
    tmpJ1 /= distance_J;
    tmpJ2 /= distance_J;
  }

  /** Compute sparse intermediary results. */
  vnl_vector< double > tmp1sparse_M = tmpM1.post_multiply( D1sparse );
  vnl_vector< double > tmp1sparse_J = tmpJ1.post_multiply( D1sparse );
  vnl_vector< double > tmp2sparse_M = tmpM2.post_multiply( D2sparse_M );
  vnl_vector< double > tmp2sparse_J = tmpJ2.post_multiply( D2sparse_J );

  /** Update dGamma_M. */
  if( distance_M > this->m_AvoidDivisionBy )
  {
    for( unsigned int i = 0; i < D1indices.size(); ++i )
    {
      dGamma_M[ D1indices[ i ] ] += tmp1sparse_M[ i ];
    }

    for( unsigned int i = 0; i < D2indices_M.size(); ++i )
    {
      dGamma_M[ D2indices_M[ i ] ] -= tmp2sparse_M[ i ];
    }
  }

  /** Update dGamma_J. */
  if( distance_J > this->m_AvoidDivisionBy )
  {
    for( unsigned int i = 0; i < D1indices.size(); ++i )
    {
      dGamma_J[ D1indices[ i ] ] += tmp1sparse_J[ i ];
    }

    for( unsigned int i = 0; i < D2indices_J.size(); ++i )
    {
      dGamma_J[ D2indices_J[ i ] ] -= tmp2sparse_J[ i ];
    }
  }

} // end UpdateDerivativeOfGammas()


/**
 * ************************ PrintSelf *************************
 */

template< class TFixedImage, class TMovingImage >
void
KNNGraphAlphaMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Alpha: " << this->m_Alpha << std::endl;
  os << indent << "AvoidDivisionBy: " << this->m_AvoidDivisionBy << std::endl;

  os << indent << "BinaryKNNTreeFixed: "
     << this->m_BinaryKNNTreeFixed.GetPointer() << std::endl;
  os << indent << "BinaryKNNTreeMoving: "
     << this->m_BinaryKNNTreeMoving.GetPointer() << std::endl;
  os << indent << "BinaryKNNTreeJoint: "
     << this->m_BinaryKNNTreeJoint.GetPointer() << std::endl;

  os << indent << "BinaryKNNTreeSearcherFixed: "
     << this->m_BinaryKNNTreeSearcherFixed.GetPointer() << std::endl;
  os << indent << "BinaryKNNTreeSearcherMoving: "
     << this->m_BinaryKNNTreeSearcherMoving.GetPointer() << std::endl;
  os << indent << "BinaryKNNTreeSearcherJoint: "
     << this->m_BinaryKNNTreeSearcherJoint.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef _itkKNNGraphAlphaMutualInformationImageToImageMetric_hxx

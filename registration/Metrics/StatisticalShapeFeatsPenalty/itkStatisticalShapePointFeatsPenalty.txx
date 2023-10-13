/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkStatisticalShapePointFeatsPenalty_txx
#define _itkStatisticalShapePointFeatsPenalty_txx

#include "itkStatisticalShapePointFeatsPenalty.h"


namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <class TFixedImage, class TMovingImage>
StatisticalShapePointFeatsPenalty<TFixedImage,TMovingImage>
::StatisticalShapePointFeatsPenalty()
{
  this->SetUseImageSampler( true );
  
  this->m_MeanVector              = NULL;
  this->m_ProposalDerivative      = NULL;
  this->m_CovarianceMatrix        = NULL;
  this->m_InverseCovarianceMatrix = NULL;

  /** Initialize the m_SSPThreaderMetricParameters. */
  this->m_SSPThreaderMetricParameters.st_Metric = this;

  // Multi-threading structs
  this->m_SSPGetValueAndDerivativePerThreadVariables     = NULL;
  this->m_SSPGetValueAndDerivativePerThreadVariablesSize = 0;

} // end Constructor()


/**
 * ************************ Destructor ***************************
 */

template <class TFixedImage, class TMovingImage>
StatisticalShapePointFeatsPenalty<TFixedImage,TMovingImage>
::~StatisticalShapePointFeatsPenalty()
{
  if ( this->m_MeanVector != NULL )
    {
    delete this->m_MeanVector;
    this->m_MeanVector = NULL;
    }
  if ( this->m_CovarianceMatrix != NULL )
    {
    delete this->m_CovarianceMatrix;
    this->m_CovarianceMatrix = NULL;
    }
  if ( this->m_ProposalDerivative != NULL )
    {
    delete this->m_ProposalDerivative;
    this->m_ProposalDerivative = NULL;
    }
  if ( this->m_InverseCovarianceMatrix != NULL )
    {
    delete this->m_InverseCovarianceMatrix;
    this->m_InverseCovarianceMatrix = NULL;
    }

  delete[] this->m_SSPGetValueAndDerivativePerThreadVariables;
    
} // end Destructor()


/**
 * ********************* Initialize *****************************
 */

template <class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage,TMovingImage>
::Initialize( void ) throw ( ExceptionObject )
{
  /** Call the superclass. */
  this->Superclass::Initialize();

  this->m_ProposalLength = this->FixedImageDimension * this->GetFixedPointSet()->GetNumberOfPoints();

  /** Automatic selection of regularization variances. */
  if ( this->m_BaseVarianceFeats == -1.0 )
    {
    VnlVectorType covDiagonal = this->m_CovarianceMatrix->get_diagonal();
    this->m_BaseVarianceFeats = covDiagonal.extract( this->m_ProposalLength ).mean();
    } // End automatic selection of regularization variances.

  VnlMatrixType regularizedCovariance = ( 1 - this->m_ShrinkageIntensityFeats ) * ( *this->m_CovarianceMatrix );
  VnlVectorType regCovDiagonal        = regularizedCovariance.get_diagonal();
  
  regCovDiagonal += this->m_ShrinkageIntensityFeats * this->m_BaseVarianceFeats;
  regularizedCovariance.set_diagonal( regCovDiagonal );
  /** If no regularization is applied, the user is responsible for providing an
   * invertible Covariance Matrix. For a Moore-Penrose pseudo inverse use
   * ShrinkageIntensity=0 and ShapeModelCalculation=1 or 2.
   */
  this->m_InverseCovarianceMatrix = new VnlMatrixType( vnl_svd_inverse( regularizedCovariance ) );
   
} // end Initialize()


/**
* ********************* InitializeSSPThreadingParameters ****************************
*/

template<class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage, TMovingImage>
::InitializeSSPThreadingParameters( void ) const
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
	if ( this->m_SSPGetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
	{
		delete[] this->m_SSPGetValueAndDerivativePerThreadVariables;
		this->m_SSPGetValueAndDerivativePerThreadVariables     = new AlignedSSPGetValueAndDerivativePerThreadStruct[ this->m_NumberOfThreads ];
		this->m_SSPGetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
	}

} // end InitializeSSPThreadingParameters()


/**
 * ************************ GetValue *************************
 */

template <class TFixedImage, class TMovingImage>
typename StatisticalShapePointFeatsPenalty<TFixedImage,TMovingImage>::MeasureType
StatisticalShapePointFeatsPenalty<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  /** Sanity checks. */
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
  if ( !fixedPointSet )
    {
    itkExceptionMacro( << "Fixed point set has not been assigned" );
    }
  
  /** Initialize some variables. */
  MeasureType value = NumericTraits< MeasureType >::Zero;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  this->m_ProposalVector.set_size( this->m_ProposalLength );

  /** Create iterators. */
  PointIterator pointItFixed = fixedPointSet->GetPoints()->Begin();
  PointIterator pointEnd     = fixedPointSet->GetPoints()->End();

  unsigned int vertexindex = 0;
  OutputPointType fixedPoint;
  /** Loop over the corresponding points. */
  while ( pointItFixed != pointEnd )
    {
    fixedPoint = pointItFixed.Value();

	this->FillProposalVector( fixedPoint, vertexindex );

    ++pointItFixed;
    vertexindex += this->FixedImageDimension;
    } // end loop over all corresponding points
  
  // TODO this declaration instantiates a zero sized vector, but it will be reassigned anyways.
  VnlVectorType differenceVector;

  this->CalculateValue( value, differenceVector );
  
  return value;

} // end GetValue()


/**
 * ************************ GetDerivative *************************
 */

template <class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage,TMovingImage>
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
 * ************************ GetValueAndDerivative *************************
 */

template <class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** Sanity checks. */
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
  if ( !fixedPointSet )
    {
    itkExceptionMacro( << "Fixed point set has not been assigned" );
    }
  
  /** Initialize some variables. */
  value = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  this->m_ProposalVector.set_size( this->m_ProposalLength );
  this->m_ProposalDerivative = new ProposalDerivativeType( this->GetNumberOfParameters(), NULL );

  /** Create iterators. */
  PointIterator pointItFixed = fixedPointSet->GetPoints()->Begin();
  PointIterator pointEnd     = fixedPointSet->GetPoints()->End();

  unsigned int vertexindex = 0;
  OutputPointType fixedPoint;
  /** Loop over the corresponding points. */
  while ( pointItFixed != pointEnd )
    {
    fixedPoint = pointItFixed.Value();
	this->FillProposalVector( fixedPoint, vertexindex );
    this->FillProposalDerivative( fixedPoint, vertexindex );

    ++pointItFixed;
    vertexindex += this->FixedImageDimension;
    } // end loop over all corresponding points
  
  // TODO this declaration instantiates a zero sized vector, but it will be reassigned anyways.
  VnlVectorType differenceVector;

  this->CalculateValue( value, differenceVector );

  if ( value != 0.0 )
    {
	/** Option for now to still use the single threaded code. */
	if ( !this->m_UseMultiThread )
	  {
	  this->CalculateDerivative( derivative, value, differenceVector );
	  }
	else
	  {
	  this->InitializeSSPThreadingParameters();
	  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
	    {
		this->m_SSPGetValueAndDerivativePerThreadVariables[ i ].st_value = value;
		this->m_SSPGetValueAndDerivativePerThreadVariables[ i ].st_differenceVector.set_size( this->m_ProposalLength );
		for ( unsigned long j = 0; j < this->m_ProposalLength; ++j )
		  {
		  this->m_SSPGetValueAndDerivativePerThreadVariables[ i ].st_differenceVector[ j ] = differenceVector[ j ];
		  }
	    }
	  this->m_SSPThreaderMetricParameters.st_DerivativePointer = derivative.begin();
	  this->m_Threader->SetNumberOfThreads( this->m_NumberOfThreads );
	  this->m_Threader->SetSingleMethod( SSPCalculateDerivativeThreaderCallback, 
		  const_cast<void *>(static_cast<const void *>(&this->m_SSPThreaderMetricParameters)) );
	  this->m_Threader->SingleMethodExecute();
	  this->AfterSSPThreadedCalculateDerivative();
	  }
    }
  else
    {
    typename ProposalDerivativeType::iterator proposalDerivativeIt  = this->m_ProposalDerivative->begin();
    typename ProposalDerivativeType::iterator proposalDerivativeEnd = this->m_ProposalDerivative->end();
    for (; proposalDerivativeIt != proposalDerivativeEnd; ++proposalDerivativeIt )
      {
      if ( *proposalDerivativeIt != NULL )
        {
        delete ( *proposalDerivativeIt );
        }
      }
    }
  delete this->m_ProposalDerivative;
  this->m_ProposalDerivative = NULL;

} // end GetValueAndDerivative()


/**
 * ******************* FillProposalVector *******************
 */

template<class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage, TMovingImage>
::FillProposalVector( const OutputPointType & fixedPoint,
  const unsigned int vertexindex ) const
{
  OutputPointType mappedPoint;
  /** Get the current corresponding points. */
  mappedPoint = this->m_AdvancedTransform->TransformPoint( fixedPoint );

  /** Copy n-D coordinates into big Shape vector. Aligning the centroids is done later. */
  for( unsigned int d = 0; d < this->FixedImageDimension; ++d )
  {
    this->m_ProposalVector[ vertexindex + d ] = mappedPoint[ d ];
  }
} // end FillProposalVector()


/**
 * ******************* FillProposalDerivative *******************
 */

template<class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage, TMovingImage>
::FillProposalDerivative( const OutputPointType & fixedPoint,
  const unsigned int vertexindex ) const
{
  /**
   * A (column) vector is constructed for each mu, only if that mu affects the shape penalty.
   * I.e. if there is at least one point of the mesh with non-zero derivatives,
   * a full column vector is instantiated (which can contain zeros for many other points)
   *
   * m_ProposalDerivative is a container with either full shape-vector-sized derivative vectors or NULL-s. Example:
   *
   * mu1: [ [ dx1/dmu1 , dy1/dmu1 , dz1/dmu1 ] , [ 0 , 0 , 0 ] , [ dx3/dmu1 , dy3/dmu1 , dz3/dmu1 ] , [...] ]^T
   * mu2: Null
   * mu3: [ [ 0 , 0 , 0 ] , [ dx2/dmu3 , dy2/dmu3 , dz2/dmu3 ] , [ dx3/dmu3 , dy3/dmu3 , dz3/dmu3 ] , [...] ]^T
   *
   */

  NonZeroJacobianIndicesType nzji(
  this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );

  /** Get the TransformJacobian dT/dmu. */
  TransformJacobianType jacobian;
  this->m_AdvancedTransform->GetJacobian( fixedPoint, jacobian, nzji );

  for( unsigned int i = 0; i < nzji.size(); ++i )
  {
    const unsigned int mu = nzji[ i ];
    if( ( *this->m_ProposalDerivative )[ mu ] == NULL )
    {
      /** Create the big column vector if it does not yet exist for this mu*/
      ( *this->m_ProposalDerivative )[ mu ] = new VnlVectorType( this->m_ProposalLength, 0.0 );
      // memory will be freed in CalculateDerivative()
    }

    /** The column vector exists for this mu, so copy the jacobians for this point into the big vector. */
    for( unsigned int d = 0; d < this->FixedImageDimension; ++d )
    {
      ( *( *this->m_ProposalDerivative )[ mu ] )[ vertexindex + d ] = jacobian.get_column( i )[ d ];
    }
  }

} // end FillProposalDerivative()


/**
 * ******************* CalculateValue *******************
 */

template<class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage, TMovingImage>
::CalculateValue( MeasureType & value,
  VnlVectorType & differenceVector ) const
{
  differenceVector = this->m_ProposalVector - *m_MeanVector;

  value = sqrt( bracket( differenceVector, *this->m_InverseCovarianceMatrix, differenceVector ) );
     
} //end CalculateValue()


/**
 * ******************* CalculateDerivative *******************
 */

template<class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage, TMovingImage>
::CalculateDerivative( DerivativeType & derivative,
  const MeasureType & value,
  const VnlVectorType & differenceVector ) const
{
  typename ProposalDerivativeType::iterator proposalDerivativeIt  = this->m_ProposalDerivative->begin();
  typename ProposalDerivativeType::iterator proposalDerivativeEnd = this->m_ProposalDerivative->end();

  typename DerivativeType::iterator derivativeIt = derivative.begin();

  for (; proposalDerivativeIt != proposalDerivativeEnd; ++proposalDerivativeIt, ++derivativeIt )
    {
    if ( *proposalDerivativeIt != NULL )
      {
      /**innerproduct diff^T * Sigma^-1 * d/dmu (diff), where iterated over mu-s*/
      *derivativeIt = bracket( differenceVector, *m_InverseCovarianceMatrix, ( **proposalDerivativeIt ) ) / value;
         
      delete ( *proposalDerivativeIt );
      }
    }

} // end CalculateDerivative()


/**
* **************** SSPCalculateDerivativeThreaderCallback *******
*/

template<class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
StatisticalShapePointFeatsPenalty<TFixedImage, TMovingImage>
::SSPCalculateDerivativeThreaderCallback( void * arg )
{
	SSPThreadInfoType * infoStruct = static_cast< SSPThreadInfoType * >( arg );
	ThreadIdType        threadID   = infoStruct->ThreadID;
	ThreadIdType       nrOfThreads = infoStruct->NumberOfThreads;

	SSPMultiThreaderParameterType * temp
		= static_cast< SSPMultiThreaderParameterType * >( infoStruct->UserData );

	const unsigned long numPar  = temp->st_Metric->GetNumberOfParameters();
	const unsigned long subSize = static_cast< unsigned int >(
		vcl_ceil( static_cast< double >( numPar )
		/ static_cast< double >( nrOfThreads ) ) );
	unsigned long jmin = threadID * subSize;
	unsigned long jmax = ( threadID + 1 ) * subSize;
	jmin = ( jmin > numPar ) ? numPar : jmin;
	jmax = ( jmax > numPar ) ? numPar : jmax;

	/** This thread accumulates all sub-derivatives into a single one, for the
	* range [ jmin, jmax [. Additionally, the sub-derivatives are reset.
	*/
	for( unsigned long mu = jmin; mu < jmax; ++mu )
	{
		if( ( *temp->st_Metric->m_ProposalDerivative )[ mu ] != NULL )
		{
			temp->st_DerivativePointer[ mu ] = bracket( temp->st_Metric->m_SSPGetValueAndDerivativePerThreadVariables[ threadID ].st_differenceVector, 
				*temp->st_Metric->m_InverseCovarianceMatrix, *( *temp->st_Metric->m_ProposalDerivative )[ mu ] ) / temp->st_Metric->m_SSPGetValueAndDerivativePerThreadVariables[ threadID ].st_value;
		}
	}

	return ITK_THREAD_RETURN_VALUE;

} // end SSPCalculateDerivativeThreaderCallback()


/**
* ********************* AfterSSPThreadedCalculateDerivative ****************************
*/

template<class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage, TMovingImage>
::AfterSSPThreadedCalculateDerivative( void ) const
{
	typename ProposalDerivativeType::iterator proposalDerivativeIt  = this->m_ProposalDerivative->begin();
	typename ProposalDerivativeType::iterator proposalDerivativeEnd = this->m_ProposalDerivative->end();
	for(; proposalDerivativeIt != proposalDerivativeEnd; ++proposalDerivativeIt )
	{
		if( *proposalDerivativeIt != NULL )
		{
			delete ( *proposalDerivativeIt );
		}
	}
	
} // end AfterSSPThreadedCalculateDerivative()


/**
 * ************************ PrintSelf *************************
 */

template <class TFixedImage, class TMovingImage>
void
StatisticalShapePointFeatsPenalty<TFixedImage,TMovingImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

} // end PrintSelf()


} // end namespace itk


#endif // end #ifndef _itkStatisticalShapePointFeatsPenalty_txx


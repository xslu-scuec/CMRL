/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxMultiMetricMultiResolutionRegistrationWithFeatures_HXX__
#define __elxMultiMetricMultiResolutionRegistrationWithFeatures_HXX__

#include "elxMultiMetricMultiResolutionRegistrationWithFeatures.h"

namespace elastix
{


  /**
   * ******************* Constructor ***********************
   */

  template <class TElastix>
    MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::MultiMetricMultiResolutionRegistrationWithFeatures()
  {
    this->m_ShowExactMetricValue = false;
  } // end constructor


  /**
   * ******************* BeforeRegistration ***********************
   */

  template <class TElastix>
    void MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::BeforeRegistration( void )
  {
    /** Get the components from this->m_Elastix and set them. */
    this->GetAndSetComponents();

    /** Set the number of resolutions. */
    unsigned int numberOfResolutions = 3;
    this->m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );
    this->SetNumberOfLevels( numberOfResolutions );

    /** Set the FixedImageRegions to the buffered regions. */
    this->GetAndSetFixedImageRegions();

    /** Set the fixed image interpolators. */
    this->GetAndSetFixedImageInterpolators();
	
	/** Add the target cells "Metric<i>" and "||Gradient<i>||" to xout["iteration"]
     * and format as floats.
     */
    const unsigned int nrOfMetrics = this->GetCombinationMultiInputMetric()->GetNumberOfMetrics();
    unsigned int width = 0;
    for ( unsigned int i = nrOfMetrics; i > 0; i /= 10 )
    {
      width++;
    }
    for ( unsigned int i = 0; i < nrOfMetrics; ++i )
    {
      std::ostringstream makestring1;
      makestring1 << "2:Metric" << std::setfill('0') << std::setw(width) << i;
      xout["iteration"].AddTargetCell( makestring1.str().c_str() );
      xl::xout["iteration"][ makestring1.str().c_str() ] << std::showpoint << std::fixed;

      std::ostringstream makestring2;
      makestring2 << "4:||Gradient" << std::setfill('0') << std::setw(width) << i << "||";
      xout["iteration"].AddTargetCell( makestring2.str().c_str() );
      xl::xout["iteration"][ makestring2.str().c_str() ] << std::showpoint << std::fixed;

      std::ostringstream makestring3;
      makestring3 << "Time" << std::setfill('0') << std::setw(width) << i << "[ms]";
      xout["iteration"].AddTargetCell( makestring3.str().c_str() );
      xl::xout["iteration"][ makestring3.str().c_str() ] << std::showpoint << std::fixed;
    }

  } // end BeforeRegistration()


  /**
   * ******************* BeforeEachResolution ***********************
   */

  template <class TElastix>
    void MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::BeforeEachResolution( void )
  {
    /** Get the current resolution level. */
    unsigned int level = this->GetCurrentLevel();

    /** Get the number of metrics. */
    unsigned int nrOfMetrics = this->GetCombinationMultiInputMetric()->GetNumberOfMetrics();
  
    /** Set the masks in the metric. */
    this->UpdateFixedMasks( level );
    this->UpdateMovingMasks( level );
	
	/** Set the metric weights. The default metric weight is 1.0 / nrOfMetrics. */
    double defaultWeight = 1.0 / static_cast<double>( nrOfMetrics );
    for ( unsigned int metricnr = 0; metricnr < nrOfMetrics; ++metricnr )
    {
      double weight = defaultWeight;
      std::ostringstream makestring;
      makestring << "Metric" << metricnr << "Weight";
      this->GetConfiguration()->ReadParameter( weight, makestring.str(), "", level, 0 );
      this->GetCombinationMultiInputMetric()->SetMetricWeight( weight, metricnr );
    }

    /** Check if the exact metric value, computed on all pixels, should be shown.
     * If at least one of the metrics has it enabled, show also the weighted sum of all
     * exact metric values. */

    /** Show the exact metric in every iteration? */
    this->m_ShowExactMetricValue = false;
    for ( unsigned int metricnr = 0; metricnr < nrOfMetrics; ++metricnr )
    {
      this->m_ShowExactMetricValue |= this->GetElastix()->
        GetElxMetricBase( metricnr )->GetShowExactMetricValue();
    }

    if ( this->m_ShowExactMetricValue )
    {
      /** Define the name of the ExactMetric column */
      std::string exactMetricColumn = "ExactMetric";

      /** Remove the ExactMetric-column, if it already existed. */
      xl::xout["iteration"].RemoveTargetCell( exactMetricColumn.c_str() );

      /** Create a new column in the iteration info table */
      xl::xout["iteration"].AddTargetCell( exactMetricColumn.c_str() );
      xl::xout["iteration"][ exactMetricColumn.c_str() ]
        << std::showpoint << std::fixed;
    }

  } // end BeforeEachResolution()


  /**
   * ******************* AfterEachIteration ***********************
   */

  template <class TElastix>
    void MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::AfterEachIteration( void )
  {
    /** Print the submetric values and gradients to xout["iteration"]. */
    const unsigned int nrOfMetrics = this->GetCombinationMultiInputMetric()->GetNumberOfMetrics();
    unsigned int width = 0;
    for ( unsigned int i = nrOfMetrics; i > 0; i /= 10 )
    {
      width++;
    }
    for ( unsigned int i = 0; i < nrOfMetrics; ++i )
    {
      std::ostringstream makestring1;
      makestring1 << "2:Metric" << std::setfill('0') << std::setw(width) << i;
      xl::xout["iteration"][ makestring1.str().c_str() ] <<
        this->GetCombinationMultiInputMetric()->GetMetricValue( i );

      std::ostringstream makestring2;
      makestring2 << "4:||Gradient" << std::setfill('0') << std::setw(width) << i << "||";
      xl::xout["iteration"][ makestring2.str().c_str() ] <<
        this->GetCombinationMultiInputMetric()->GetMetricDerivativeMagnitude( i );

      std::ostringstream makestring3;
      makestring3 << "Time" << std::setfill('0') << std::setw(width) << i << "[ms]";
      xl::xout["iteration"][ makestring3.str().c_str() ] <<
        this->GetCombinationMultiInputMetric()->GetMetricComputationTime( i );
    }

    if ( this->m_ShowExactMetricValue )
    {
      double currentExactMetricValue = 0.0;

      for ( unsigned int i = 0; i < nrOfMetrics; ++i )
      {
        const double currentExactMetricValue_i = this->GetElastix()->
            GetElxMetricBase( i )->GetCurrentExactMetricValue();

        const double weight_i = this->GetCombinationMultiInputMetric()->GetMetricWeight( i );

        currentExactMetricValue += weight_i * currentExactMetricValue_i;
      }

      xl::xout["iteration"][ "ExactMetric" ] << currentExactMetricValue;
    }

  } // end AfterEachIteration()


  /**
   * *********************** GetAndSetComponents ************************
   */

  template <class TElastix>
    void MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::GetAndSetComponents( void )
  {
    /** Get the component from this->GetElastix() (as elx::...BaseType *),
     * cast it to the appropriate type and set it in 'this'.
     */

    /** Set the metric. */
    const unsigned int nrOfMetrics = this->GetElastix()->GetNumberOfMetrics();
    this->GetCombinationMultiInputMetric()->SetNumberOfMetrics( nrOfMetrics );
    for ( unsigned int i = 0; i < nrOfMetrics; ++i )
    {
      this->GetCombinationMultiInputMetric()->SetMetric( this->GetElastix()->
        GetElxMetricBase( i )->GetAsITKBaseType(), i );
    }

    /** Set the fixed images. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImages(); ++i )
    {
      this->SetFixedImage( this->GetElastix()->GetFixedImage( i ), i );
    }

    /** Set the moving images. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfMovingImages(); ++i )
    {
      this->SetMovingImage( this->GetElastix()->GetMovingImage( i ), i );
    }

    /** Set the fixed image pyramids. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImagePyramids(); ++i )
    {
      this->SetFixedImagePyramid( this->GetElastix()->
        GetElxFixedImagePyramidBase( i )->GetAsITKBaseType(), i );
    }

    /** Set the moving image pyramids. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfMovingImagePyramids(); ++i )
    {
      this->SetMovingImagePyramid( this->GetElastix()->
        GetElxMovingImagePyramidBase( i )->GetAsITKBaseType(), i );
    }

    /** Set the moving image interpolators. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfInterpolators(); ++i )
    {
      this->SetInterpolator( this->GetElastix()->
        GetElxInterpolatorBase( i )->GetAsITKBaseType(), i );
    }

    /** Set the optimizer. */
    this->SetOptimizer( dynamic_cast<OptimizerType*>(
      this->GetElastix()->GetElxOptimizerBase()->GetAsITKBaseType() ) );

    /** Set the transform. */
    this->SetTransform( this->GetElastix()->
      GetElxTransformBase()->GetAsITKBaseType() );

    /** Samplers are not always needed: */
	for ( unsigned int i = 0; i < nrOfMetrics; ++i )
    {
      if ( this->GetElastix()->GetElxMetricBase( i )->GetAdvancedMetricUseImageSampler() )
      {
        /** Try the i-th sampler for the i-th metric. */
        if ( this->GetElastix()->GetElxImageSamplerBase( i ) )
        {
          this->GetElastix()->GetElxMetricBase( i )->SetAdvancedMetricImageSampler(
            this->GetElastix()->GetElxImageSamplerBase( i )->GetAsITKBaseType() );
        }
        else
        {
          /** Try the zeroth image sampler for each metric. */
          if ( this->GetElastix()->GetElxImageSamplerBase( 0 ) )
          {
            this->GetElastix()->GetElxMetricBase( i )->SetAdvancedMetricImageSampler(
              this->GetElastix()->GetElxImageSamplerBase( 0 )->GetAsITKBaseType() );
          }
          else
          {
            xl::xout["error"] << "ERROR: No ImageSampler has been specified." << std::endl;
            itkExceptionMacro( << "One of the metrics requires an ImageSampler, but it is not available!" );
          }
        }
      } // if sampler required by metric
    } // for loop over metrics

  } // end GetAndSetComponents()


  /**
   * *********************** GetAndSetFixedImageRegions ************************
   */

  template <class TElastix>
    void MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::GetAndSetFixedImageRegions( void )
  {
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImages(); ++i )
    {
      /** Make sure the fixed image is up to date. */
      try
      {
        this->GetElastix()->GetFixedImage( i )->Update();
      }
      catch( itk::ExceptionObject & excp )
      {
        /** Add information to the exception. */
        excp.SetLocation( "MultiResolutionRegistrationWithFeatures - BeforeRegistration()" );
        std::string err_str = excp.GetDescription();
        err_str += "\nError occured while updating region info of the fixed image.\n";
        excp.SetDescription( err_str );
        /** Pass the exception to an higher level. */
        throw excp;
      }

      /** Set the fixed image region. */
      this->SetFixedImageRegion( this->GetElastix()->GetFixedImage( i )->GetBufferedRegion(), i );
    }

  } // end GetAndSetFixedImageRegions()


  /**
   * *********************** GetAndSetFixedImageInterpolators ************************
   */

  template <class TElastix>
    void MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::GetAndSetFixedImageInterpolators( void )
  {
    /** Short cut. */
    const unsigned int noFixIm = this->GetNumberOfFixedImages();

    /** Get the spline order of the fixed feature image interpolators. */
    unsigned int splineOrder = 1;
    this->m_Configuration->ReadParameter(
      splineOrder, "FixedImageInterpolatorBSplineOrder", 0 );
    std::vector< unsigned int > soFII( noFixIm, splineOrder );
    for ( unsigned int i = 1; i < noFixIm; ++i )
    {
      this->m_Configuration->ReadParameter(
        soFII[ i ], "FixedImageInterpolatorBSplineOrder", i, false );
    }

    /** Create and set interpolators for the fixed feature images. */
    typedef itk::BSplineInterpolateImageFunction< FixedImageType >             FixedImageInterpolatorType;
    typedef std::vector< typename FixedImageInterpolatorType::Pointer >   FixedImageInterpolatorVectorType;
    FixedImageInterpolatorVectorType interpolators( noFixIm );
    for ( unsigned int i = 0; i < noFixIm; i++ )
    {
      interpolators[ i ] = FixedImageInterpolatorType::New();
      interpolators[ i ]->SetSplineOrder( soFII[ i ] );
      this->SetFixedImageInterpolator( interpolators[ i ], i );
    }

  } // end GetAndSetFixedImageInterpolators()


  /**
   * ************************* UpdateFixedMasks ************************
   */

  template <class TElastix>
    void MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::UpdateFixedMasks( unsigned int level )
  {
    /** Use only one mask. */
    const unsigned int nrOfFixedImageMasks = 1;

    /** Array of bools, that remembers for each mask if erosion is wanted. */
    UseMaskErosionArrayType useMaskErosionArray;

    /** Bool that remembers if mask erosion is wanted in any of the masks
     * remains false when no masks are used.
     */
    bool useMaskErosion;

    /** Read whether mask erosion is wanted, if any masks were supplied. */
    useMaskErosion = this->ReadMaskParameters( useMaskErosionArray,
      nrOfFixedImageMasks, "Fixed", level );

    /** Create and start timer, to time the whole mask configuration procedure. */
    itk::TimeProbe timer;
    timer.Start();

    /** Set the fixed image mask. Only one mask is assumed here. */
    FixedMaskSpatialObjectPointer fixedMask = this->GenerateFixedMaskSpatialObject(
      this->GetElastix()->GetFixedMask(), useMaskErosion,
      this->GetFixedImagePyramid(), level );
    this->GetCombinationMultiInputMetric()->SetFixedImageMask( fixedMask, 0 );

    /** Stop timer and print the elapsed time. */
	timer.Stop();
    elxout << "Setting the fixed masks took: "
		<< static_cast<long>( timer.GetMean() * 1000 )
      << " ms." << std::endl;
	  
  } // end UpdateFixedMasks()


  /**
   * ************************* UpdateMovingMasks ************************
   */

  template <class TElastix>
    void MultiMetricMultiResolutionRegistrationWithFeatures<TElastix>
    ::UpdateMovingMasks( unsigned int level )
  {
    /** Use only one mask. */
    const unsigned int nrOfMovingImageMasks = 1;

    /** Array of bools, that remembers for each mask if erosion is wanted. */
    UseMaskErosionArrayType useMaskErosionArray;

    /** Bool that remembers if mask erosion is wanted in any of the masks
     * remains false when no masks are used.
     */
    bool useMaskErosion;

    /** Read whether mask erosion is wanted, if any masks were supplied. */
    useMaskErosion = this->ReadMaskParameters( useMaskErosionArray,
      nrOfMovingImageMasks, "Moving", level );

    /** Create and start timer, to time the whole mask configuration procedure. */
    itk::TimeProbe timer;
    timer.Start();

    /** Set the moving image mask. Only one mask is assumed here. */
    MovingMaskSpatialObjectPointer movingMask = this->GenerateMovingMaskSpatialObject(
      this->GetElastix()->GetMovingMask(), useMaskErosion,
      this->GetMovingImagePyramid(), level );
    this->GetCombinationMultiInputMetric()->SetMovingImageMask( movingMask, 0 );

    /** Stop timer and print the elapsed time. */
	timer.Stop();
    elxout << "Setting the moving masks took: "
		<< static_cast<long>( timer.GetMean() * 1000 )
      << " ms." << std::endl;
	
  } // end UpdateMovingMasks()


} // end namespace elastix

#endif // end #ifndef __elxMultiMetricMultiResolutionRegistrationWithFeatures_HXX__

